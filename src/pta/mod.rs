// Copyright (c) 2024 <Wei Li>.
//
// This source code is licensed under the GNU license found in the
// LICENSE file in the root directory of this source tree.

use log::*;
use std::time::Instant;

use rustc_driver::Compilation;
use rustc_interface::{interface, Queries};
use rustc_middle::ty::TyCtxt;

use self::andersen::AndersenPTA;
use self::context_sensitive::ContextSensitivePTA;
use self::strategies::context_strategy::KCallSiteSensitive;
use crate::graph::pag::*;
use crate::mir::function::FuncId;
use crate::mir::analysis_context::AnalysisContext;
use crate::pts_set::points_to::HybridPointsToSet;
use crate::pts_set::pt_data::DiffPTData;
use crate::util::mem_watcher::MemoryWatcher;
use crate::util::options::AnalysisOptions;
use crate::pta::result::PointerAnalysisResult;

pub mod andersen;
pub mod context_sensitive;
pub mod propagator;

pub mod result;
pub mod strategies;

pub type NodeId = PAGNodeId;
pub type EdgeId = PAGEdgeId;
pub type PointsTo<T> = HybridPointsToSet<T>;
pub type DiffPTDataTy = DiffPTData<NodeId, NodeId, PointsTo<NodeId>>;

#[derive(Clone, Copy, Debug)]
pub enum PTAType {
    Andersen,
    CallSiteSensitive
}

pub trait PointerAnalysis<'pta, 'tcx, 'compilation> {
    fn pre_analysis(&mut self) {}
    // Initialization for the analysis.
    fn initialize(&mut self);
    // Solve the worklist problem.
    fn propagate(&mut self);
    // Finalize the analysis.
    fn finalize(&self);

    fn analyze(&mut self) {
        self.pre_analysis();

        // Main analysis phase
        let now = Instant::now();

        self.initialize();
        self.propagate();
        
        let elapsed = now.elapsed();
        println!("Pointer analysis completed.");
        println!(
            "Analysis time: {}",
            humantime::format_duration(elapsed).to_string()
        );

        self.finalize();
    }

    fn get_result(&mut self) -> PointerAnalysisResult<'tcx, 'compilation>;
}

pub struct PTACallbacks {
    /// Options provided to the analysis.
    pub options: AnalysisOptions,
    /// The relative path of the file being compiled.
    file_name: String,
}

/// Constructor
impl PTACallbacks {
    pub fn new(options: AnalysisOptions) -> PTACallbacks {
        PTACallbacks {
            options,
            file_name: String::new(),
        }
    }

    pub fn run_pointer_analysis<'pta, 'tcx, 'compilation>(&mut self, compiler: &'compilation interface::Compiler, tcx: TyCtxt<'tcx>) -> PointerAnalysisResult<'tcx, 'compilation> {
        let mut mem_watcher = MemoryWatcher::new();
        mem_watcher.start();

        let result = if let Some(mut acx) = AnalysisContext::new(&compiler.sess, tcx, self.options.clone()) {
            let mut pta: Box<dyn PointerAnalysis> = match self.options.pta_type {
                PTAType::CallSiteSensitive => {
                    Box::new(
                        ContextSensitivePTA::new(
                            &mut acx, 
                            KCallSiteSensitive::new(self.options.context_depth as usize)
                        ),
                    )
                }
                PTAType::Andersen => Box::new(AndersenPTA::new(&mut acx)),
            };
            pta.analyze();
            pta.get_result()
        } else {
            panic!("AnalysisContext Initialization Failed");
        };

        mem_watcher.stop();
        result
    }

}

impl rustc_driver::Callbacks for PTACallbacks {
    /// Called before creating the compiler instance
    fn config(&mut self, config: &mut interface::Config) {
        self.file_name = config.input.source_name().prefer_remapped_unconditionaly().to_string();
        debug!("Processing input file: {}", self.file_name);
    }

    /// Called after the compiler has completed all analysis passes and before it lowers MIR to LLVM IR.
    /// At this point the compiler is ready to tell us all it knows and we can proceed to do abstract
    /// interpretation of all of the functions that will end up in the compiler output.
    /// If this method returns false, the compilation will stop.
    fn after_analysis<'tcx>(
        &mut self,
        compiler: &interface::Compiler,
        queries: &'tcx Queries<'tcx>,
    ) -> Compilation {
        compiler.sess.dcx().abort_if_errors();
        queries
            .global_ctxt()
            .unwrap()
            .enter(|tcx| self.run_pointer_analysis(compiler, tcx));
        Compilation::Continue
    }
}

