//! This is the result of pointer analysis.
//! It contains
//! - Call graph. Note that nodes in the call graph are functions,
//!   without any information about the call sites.
//! - Pointer assignment graph.

use log::*;
use petgraph::{
    graph::{DefaultIx, EdgeIndex, NodeIndex},
    Graph,
};
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_middle::mir::Location;
use rustc_middle::ty::TyCtxt;
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    rc::Rc,
};

use crate::{
    graph::call_graph::{CGCallSite, CGFunction, CallGraph as PTACallGraph},
    mir::{
        analysis_context::AnalysisContext,
        call_site::BaseCallSite,
        function::FuncId,
        path::{PathEnum, ProjectionElems},
    },
};

pub struct PointerAnalysisResult<'tcx, 'compilation> {
    pub call_graph: CallGraph,
    pub analysis_context: AnalysisContext<'tcx, 'compilation>,
    phantom_data: std::marker::PhantomData<&'tcx ()>,
}

/// A CallGraph is a graph that has functions as nodes and call sites as edges.
pub type CallGraph = MyGraph<Func, Location>;

type MyGraphNodeId = NodeIndex<DefaultIx>;
#[derive(Clone, Debug)]
pub struct MyGraph<V, E> {
    pub graph: Graph<V, E>,
    pub node_id_map: HashMap<V, MyGraphNodeId>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Func {
    pub krate: CrateNum,
    pub def_id: DefId,
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub enum Path {
    /// Locals [arg_count+1..] are the local variables and compiler temporaries.
    LocalVariable {
        func: DefId,
        ordinal: usize,
    },

    /// Locals [1..=arg_count] are the parameters
    Parameter {
        func: DefId,
        ordinal: usize,
    },

    /// Local 0 is the return value temporary
    ReturnValue {
        func: DefId,
    },

    /// Auxiliary local variable created when running pointer analysis
    Auxiliary {
        func: DefId,
        ordinal: usize,
    },

    /// A dynamically allocated memory object.
    HeapObj {
        func: DefId,
        location: Location,
    },

    /// This path points to data that is not used, but exists only to satisfy a static checker
    /// that a generic parameter is actually used.
    // PhantomData,
    Constant,

    StaticVariable {
        func: DefId,
    },

    /// The ordinal is an index into a method level table of MIR bodies.
    PromotedConstant {
        def_id: DefId,
        ordinal: usize,
    },

    /// The base denotes some struct, collection or heap_obj.
    /// projection: a non-empty list of projections
    QualifiedPath {
        base: Rc<Path>,
        projection: ProjectionElems,
    },

    OffsetPath {
        base: Rc<Path>,
        offset: usize,
    },

    /// A function instance which can be pointed to by a function pointer.
    Function(DefId),

    PromotedStrRefArray,

    PromotedArgumentV1Array,

    /// A type instance uniquely identified by the type's index in type cache
    Type(usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EdgeProvenance(pub DefId, pub Option<Location>);

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum DirectEdgeKind {
    Call,
    Ret,
    Direct,
}

// #[derive(Clone, Debug, PartialEq)]
// pub enum PAGEdgeEnum {
//     /// Statements that create a reference or a raw pointer to a place.
//     AddrPAGEdge(EdgeProvenance),
//     /// Statements that create a value by direct assignment, including Move and Copy statements.
//     DirectPAGEdge(EdgeProvenance, DirectEdgeKind),
//     /// Statements that create a value by loading the value pointed by a pointer.
//     /// e.g. `_2 = (*_3), _2 = (*_3).0.1`.
//     LoadPAGEdge(ProjectionElems, EdgeProvenance),
//     /// Statements that store a value to a pointer's pointee.
//     /// e.g. `(*_1) = _2, (*_1).0.1 = _2`.
//     StorePAGEdge(ProjectionElems, EdgeProvenance),
//     /// Similar to GetElementPtr instruction in llvm ir, get an element's address from
//     /// a pointed-to object, e.g. `_2 = &((*_3).0.1)`
//     GepPAGEdge(ProjectionElems, EdgeProvenance),
//     /// Cast a  pointer to another type
//     CastPAGEdge(EdgeProvenance),
//     /// Statements that offset a pointer.
//     OffsetPAGEdge(EdgeProvenance),
//     /// Projection, a.b -> a
//     ProjectionPAGEdge(EdgeProvenance),
// }

// impl PAGEdgeEnum {
//     pub fn to_string_with_tcx(&self, tcx: TyCtxt) -> String {
//         use PAGEdgeEnum::*;
//         match self {
//             AddrPAGEdge(prov) => format!("Addr: {}", prov.to_string_with_tcx(tcx)),
//             DirectPAGEdge(prov, kind) => format!("Direct: {} {:?}", prov.to_string_with_tcx(tcx), kind),
//             LoadPAGEdge(proj, prov) => format!("Load: {}", prov.to_string_with_tcx(tcx)),
//             StorePAGEdge(proj, prov) => format!("Store: {}", prov.to_string_with_tcx(tcx)),
//             GepPAGEdge(proj, prov) => format!("Gep: {}", prov.to_string_with_tcx(tcx)),
//             CastPAGEdge(prov) => format!("Cast: {}", prov.to_string_with_tcx(tcx)),
//             OffsetPAGEdge(prov) => format!("Offset: {}", prov.to_string_with_tcx(tcx)),
//             ProjectionPAGEdge(prov) => format!("Projection: {}", prov.to_string_with_tcx(tcx)),
//         }
//     }
// }

impl EdgeProvenance {
    pub fn to_string_with_tcx(&self, tcx: TyCtxt) -> String {
        let body = tcx.optimized_mir(self.0);
        match self.1 {
            Some(location) => {
                let block = &(body.basic_blocks[location.block]);
                if block.statements.len() == location.statement_index {
                    let ter = block.terminator();
                    format!("{:?} ||| {:?} @ {:?}", ter, self.0, ter.source_info.span)
                } else {
                    let stmt = block.statements.get(location.statement_index).unwrap();
                    format!("{:?} ||| {:?} @ {:?}", stmt, self.0, stmt.source_info.span)
                }
            }
            None => format!("{:?}", self.0),
        }
    }
}

// impl From<pag::DirectEdgeKind> for DirectEdgeKind {
//     fn from(kind: pag::DirectEdgeKind) -> Self {
//         use pag::DirectEdgeKind::*;
//         match kind {
//             Call => DirectEdgeKind::Call,
//             Ret => DirectEdgeKind::Ret,
//             Direct => DirectEdgeKind::Direct,
//         }
//     }
// }

impl Path {
    pub fn def_id(&self) -> Option<DefId> {
        use Path::*;
        match self {
            LocalVariable { func, .. }
            | Parameter { func, .. }
            | ReturnValue { func }
            | Auxiliary { func, .. }
            | HeapObj { func, .. }
            | StaticVariable { func } => Some(*func),
            QualifiedPath { base, .. } => base.def_id(),
            OffsetPath { base, .. } => base.def_id(),
            Constant
            | Function(..)
            | PromotedConstant { .. }
            | PromotedArgumentV1Array
            | PromotedStrRefArray
            | Type(..) => None,
        }
    }

    pub fn get_containing_func(&self) -> Option<DefId> {
        use Path::*;
        match self {
            LocalVariable { func, .. }
            | Parameter { func, .. }
            | ReturnValue { func }
            | Auxiliary { func, .. }
            | HeapObj { func, .. } => Some(*func),
            QualifiedPath { base, .. } | OffsetPath { base, .. } => base.get_containing_func(),
            Constant
            | StaticVariable { .. }
            | PromotedConstant { .. }
            | Function(..)
            | PromotedArgumentV1Array
            | PromotedStrRefArray
            | Type(..) => None,
        }
    }
    /// Creates a path to the local variable corresponding to the ordinal.
    pub fn new_local(func: DefId, ordinal: usize) -> Rc<Path> {
        Rc::new(Path::LocalVariable { func, ordinal })
    }

    /// Creates a path to the parameter corresponding to the ordinal.
    pub fn new_parameter(func: DefId, ordinal: usize) -> Rc<Path> {
        Rc::new(Path::Parameter { func, ordinal })
    }

    /// Creates a path to the return value.
    pub fn new_return_value(func: DefId) -> Rc<Path> {
        Rc::new(Path::ReturnValue { func })
    }

    /// Creates a path to the local variable, parameter or result local, corresponding to the ordinal.
    pub fn new_local_parameter_or_result(func: DefId, ordinal: usize, argument_count: usize) -> Rc<Path> {
        if ordinal == 0 {
            Self::new_return_value(func)
        } else if ordinal <= argument_count {
            Self::new_parameter(func, ordinal)
        } else {
            Self::new_local(func, ordinal)
        }
    }

    /// Creates a new auxiliary path.
    pub fn new_aux(func: DefId, ordinal: usize) -> Rc<Path> {
        Rc::new(Path::Auxiliary { func, ordinal })
    }

    // /// Creates a path to the heap object.
    // pub fn new_heap_obj(func_id: FuncId, location: Location) -> Rc<Path> {
    //     Rc::new(Path {
    //         value: PathEnum::HeapObj { func_id, location },
    //     })
    // }

    // /// Creates a path to a constant.
    // pub fn new_constant() -> Rc<Path> {
    //     Rc::new(Path {
    //         value: PathEnum::Constant,
    //     })
    // }

    // /// Creates a path to a static variable.
    // pub fn new_static_variable(def_id: DefId) -> Rc<Path> {
    //     Rc::new(Path {
    //         value: PathEnum::StaticVariable { def_id },
    //     })
    // }

    // /// Creates a path to a promoted constant.
    // pub fn new_promoted(def_id: DefId, ordinal: usize) -> Rc<Path> {
    //     Rc::new(Path {
    //         value: PathEnum::PromotedConstant { def_id, ordinal },
    //     })
    // }

    // /// Creates a path to a argumentv1 array.
    // pub fn new_argumentv1_arr() -> Rc<Path> {
    //     Rc::new(Path {
    //         value: PathEnum::PromotedArgumentV1Array,
    //     })
    // }

    // /// Creates a path to a &str array.
    // pub fn new_str_ref_arr() -> Rc<Path> {
    //     Rc::new(Path {
    //         value: PathEnum::PromotedStrRefArray,
    //     })
    // }

    /// Creates a path that qualifies the given root path with the given projection.
    pub fn new_qualified(base: Rc<Path>, projection: ProjectionElems) -> Rc<Path> {
        assert!(!matches!(*base, Path::QualifiedPath { .. }));
        Rc::new(Path::QualifiedPath { base, projection })
    }

    // /// Creates a path that qualifies the given root path with the given offset.
    // pub fn new_offset(base: Rc<Path>, offset: usize) -> Rc<Path> {
    //     if offset == 0 {
    //         base
    //     } else {
    //         Rc::new(Path {
    //             value: PathEnum::OffsetPath { base, offset },
    //         })
    //     }
    // }

    // /// Creates a path that selects the element at a given index value of the array at the given path.
    // pub fn new_index(collection_path: Rc<Path>) -> Rc<Path> {
    //     Path::append_projection_elem(&collection_path, PathSelector::Index)
    // }

    // /// Creates a path that selects the given field of the struct at the given path.
    // pub fn new_field(base: Rc<Path>, field_index: usize) -> Rc<Path> {
    //     Self::append_projection_elem(&base, PathSelector::Field(field_index))
    // }

    // /// Creates a path that selects the given union field of the union at the given path.
    // pub fn new_union_field(base: Rc<Path>, field_index: usize) -> Rc<Path> {
    //     Self::append_projection_elem(&base, PathSelector::UnionField(field_index))
    // }

    // /// Creates a path that selects the given downcast of the enum at the given path.
    // pub fn new_downcast(base: Rc<Path>, downcast_variant: usize) -> Rc<Path> {
    //     Self::append_projection_elem(&base, PathSelector::Downcast(downcast_variant))
    // }

    // /// Creates a path referring to function item.
    // pub fn new_function(func_id: FuncId) -> Rc<Path> {
    //     Rc::new(Path {
    //         value: PathEnum::Function(func_id),
    //     })
    // }

    // /// Creates a path referring to a type item.
    // pub fn new_type(index: usize) -> Rc<Path> {
    //     Rc::new(Path {
    //         value: PathEnum::Type(index),
    //     })
    // }

    // /// Creates a path to the target memory of a reference value.
    // pub fn new_deref(address_path: Rc<Path>) -> Rc<Path> {
    //     assert!(!matches!(address_path.value, PathEnum::QualifiedPath { .. }));
    //     Rc::new(Path {
    //         value: PathEnum::QualifiedPath {
    //             base: address_path,
    //             projection: vec![PathSelector::Deref],
    //         },
    //     })
    // }

    // /// Creates a path representing the metadata of a dynamic pointer.
    // pub fn dyn_ptr_metadata(dyn_ptr_path: &Rc<Path>) -> Rc<Path> {
    //     Path::add_offset(dyn_ptr_path, PTR_METADATA_OFFSET)
    // }

    // /// Creates a path by appending the projection elem.
    // pub fn append_projection_elem(path: &Rc<Path>, projection_elem: PathSelector) -> Rc<Path> {
    //     match &path.value {
    //         PathEnum::QualifiedPath { base, projection } => {
    //             let mut projection = projection.clone();
    //             projection.push(projection_elem);
    //             Path::new_qualified(base.clone(), projection)
    //         }
    //         _ => Path::new_qualified(path.clone(), vec![projection_elem]),
    //     }
    // }

    // /// Creates a path by appending the projection elems.
    // pub fn append_projection(path: &Rc<Path>, projection_elems: &ProjectionElems) -> Rc<Path> {
    //     if projection_elems.is_empty() {
    //         return path.clone();
    //     }
    //     match &path.value {
    //         PathEnum::QualifiedPath { base, projection } => {
    //             let mut projection = projection.clone();
    //             projection.extend_from_slice(projection_elems);
    //             Path::new_qualified(base.clone(), projection)
    //         }
    //         _ => Path::new_qualified(path.clone(), projection_elems.clone()),
    //     }
    // }

    // pub fn add_offset(path: &Rc<Path>, offset: usize) -> Rc<Path> {
    //     if offset == 0 {
    //         return path.clone();
    //     }
    //     match &path.value {
    //         PathEnum::OffsetPath {
    //             base,
    //             offset: old_offset,
    //         } => Path::new_offset(base.clone(), old_offset + offset),
    //         _ => {
    //             if let PathEnum::QualifiedPath { base: _, projection } = &path.value {
    //                 assert!(projection.len() == 1 && projection[0] == PathSelector::Deref);
    //             }
    //             Path::new_offset(path.clone(), offset)
    //         }
    //     }
    // }

    // /// Creates a path by truncating the projection elems.
    // pub fn truncate_projection_elems(path: &Rc<Path>, len: usize) -> Rc<Path> {
    //     if let PathEnum::QualifiedPath { base, projection } = &path.value {
    //         if projection.len() < len {
    //             warn!("The given length is langer than the projection elements length.");
    //             path.clone()
    //         } else {
    //             if len == 0 {
    //                 base.clone()
    //             } else {
    //                 Path::new_qualified(base.clone(), projection[..len].to_vec())
    //             }
    //         }
    //     } else {
    //         warn!("Truncating a non-qualified path");
    //         path.clone()
    //     }
    // }

    // /// Returns the original path by removing the cast.
    // pub fn remove_cast(path: &Rc<Path>) -> Rc<Path> {
    //     if let PathEnum::QualifiedPath { base: _, projection } = &path.value {
    //         if let PathSelector::Cast(_) = projection.last().unwrap() {
    //             Path::truncate_projection_elems(&path, projection.len() - 1)
    //         } else {
    //             path.clone()
    //         }
    //     } else {
    //         path.clone()
    //     }
    // }

    // pub fn is_constant(&self) -> bool {
    //     matches!(self.value, PathEnum::Constant)
    // }

    // pub fn is_promoted_constant(&self) -> bool {
    //     matches!(self.value, PathEnum::PromotedConstant { .. })
    // }

    // pub fn is_static_variable(&self) -> bool {
    //     matches!(self.value, PathEnum::StaticVariable { .. })
    // }
}

// impl<'tcx> DataflowGraph<'tcx> {
//     pub fn graph(&self) -> &Graph<Rc<Path>, PAGEdgeEnum> {
//         &self.graph.graph
//     }
// }

impl CallGraph {
    pub fn entry_point(&self) -> Option<Func> {
        // Find the only node with no incoming edges
        let mut entry_point = HashSet::new();
        for node in self.graph.node_indices() {
            if self
                .graph
                .edges_directed(node, petgraph::Direction::Incoming)
                .count()
                == 0
            {
                entry_point.insert(self.graph[node].clone());
            }
        }
        if entry_point.len() == 1 {
            Some(entry_point.into_iter().next().unwrap())
        } else {
            panic!(
                "There are multiple entry points in the call graph: {:?}",
                entry_point
            );
        }
    }
}

impl<'pta, 'tcx, 'compilation> PointerAnalysisResult<'tcx, 'compilation> {
    pub fn new<F, S>(
        acx: &'pta mut AnalysisContext<'tcx, 'compilation>,
        call_graph: &PTACallGraph<F, S>,
    ) -> PointerAnalysisResult<'tcx, 'compilation>
    where
        F: CGFunction + Into<FuncId>,
        S: CGCallSite + Into<BaseCallSite>,
    {
        PointerAnalysisResult {
            call_graph: extract_call_graph(acx, call_graph),
            analysis_context: acx.clone(),
            phantom_data: std::marker::PhantomData,
        }
    }
}

#[allow(dead_code)]
impl<V, E> MyGraph<V, E>
where
    V: std::hash::Hash + Eq + Clone,
{
    fn new() -> Self {
        MyGraph {
            graph: Graph::new(),
            node_id_map: HashMap::new(),
        }
    }

    fn add_edge(&mut self, e: E, from: V, to: V) {
        let from_id = self.get_or_insert_node(from);
        let to_id = self.get_or_insert_node(to);
        self.graph.add_edge(from_id, to_id, e);
    }

    fn get_or_insert_node(&mut self, func: V) -> MyGraphNodeId {
        if let Some(node_id) = self.node_id_map.get(&func) {
            return *node_id;
        }
        let node_id = self.graph.add_node(func.clone());
        self.node_id_map.insert(func, node_id);
        node_id
    }

    pub fn iter_nodes(&self) -> impl Iterator<Item = &V> {
        self.graph.node_indices().map(move |node_id| &self.graph[node_id])
    }

    pub fn iter_edges(&self) -> impl Iterator<Item = (MyGraphNodeId, MyGraphNodeId, &E)> {
        self.graph.edge_indices().map(move |edge_id| {
            let (from_id, to_id) = self.graph.edge_endpoints(edge_id).unwrap();
            (from_id, to_id, &self.graph[edge_id])
        })
    }

    pub fn node_data(&self, node_id: MyGraphNodeId) -> &V {
        &self.graph[node_id]
    }

    pub fn edge_data(&self, edge_id: EdgeIndex) -> &E {
        &self.graph[edge_id]
    }
}

fn extract_call_graph<F, S>(acx: &AnalysisContext, call_graph: &PTACallGraph<F, S>) -> CallGraph
where
    F: CGFunction + Into<FuncId>,
    S: CGCallSite + Into<BaseCallSite>,
{
    let mut ret = CallGraph::new();
    let get_def_id = |func_id: NodeIndex| -> DefId {
        func_id_to_def_id(acx, call_graph.graph.node_weight(func_id).unwrap().func)
    };

    for (callsite, edges) in &call_graph.callsite_to_edges {
        let ci_callsite: BaseCallSite = callsite.clone().into();
        for edge in edges {
            let (from_id, to_id) = call_graph.graph.edge_endpoints(*edge).unwrap();
            let from_func = get_def_id(from_id);
            let to_func = get_def_id(to_id);
            ret.add_edge(ci_callsite.location, from_func.into(), to_func.into());
        }
    }
    ret
}

// fn extract_dataflow_graph<'tcx, P: PAGPath>(
//     acx: &mut AnalysisContext<'tcx, '_>,
//     pag: &PAG<P>,
// ) -> DataflowGraph<'tcx> {
//     let mut path_ty: HashMap<Rc<Path>, HashSet<Ty>> = HashMap::new();
//     let mut graph = MyGraph::new();
//     // let mut update_path_ty = |path: &Path, ty: Ty|
//     for edge in pag.graph().edge_references() {
//         let src_origin = pag.node_path(edge.source()); //.regularize(acx);
//         let src = root_path(convert_path(acx, src_origin.value()));
//         let dst_origin = pag.node_path(edge.target()); //.regularize(acx); TODO: why regularize causes an error?
//         let dst = root_path(convert_path(acx, dst_origin.value()));
//         graph.add_edge(convert_edge(acx, &edge.weight().kind), src.clone(), dst.clone());
//         let src_ty = mir_root_type(acx, src_origin.value());
//         match path_ty.get_mut(&src) {
//             Some(ty_set) => {
//                 ty_set.insert(src_ty);
//             }
//             None => {
//                 let mut ty_set = HashSet::new();
//                 ty_set.insert(src_ty);
//                 path_ty.insert(src.clone(), ty_set);
//             }
//         };
//         let dst_ty = mir_root_type(acx, dst_origin.value());
//         match path_ty.get_mut(&dst) {
//             Some(ty_set) => {
//                 ty_set.insert(dst_ty);
//             }
//             None => {
//                 let mut ty_set = HashSet::new();
//                 ty_set.insert(dst_ty.clone());
//                 path_ty.insert(dst.clone(), ty_set);
//             }
//         };
//     }
//     DataflowGraph { graph, path_ty }
// }

#[allow(dead_code)]
fn print_node(acx: &AnalysisContext, node: &PathEnum) {
    use PathEnum::*;
    match node {
        Parameter { func_id, ordinal } => {
            let func = func_id_to_def_id(acx, *func_id);
            let func_name = acx.tcx.def_path_str(func);
            if func_name.contains("transpose") {
                debug!("Found Parameter in PAG: {:?}; ordinal: {}", func, ordinal);
            }
        }
        _ => {}
    }
}

impl From<DefId> for Func {
    fn from(def_id: DefId) -> Self {
        Func {
            krate: def_id.krate,
            def_id,
        }
    }
}

fn func_id_to_def_id<F>(acx: &AnalysisContext, func_id: F) -> DefId
where
    F: CGFunction + Into<FuncId>,
{
    let func_id = func_id.into();
    acx.functions
        .get(func_id)
        .expect("Failed to get the given func_id in the analysis context")
        .def_id
}
