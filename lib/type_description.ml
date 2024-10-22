open Ctypes

module Types (F : TYPE) = struct
  open F

  type session

  type error

  type source

  type output

  type invocation

  type pipeline = Std | Hal_executable | Precompile

  let session : session structure typ = structure "iree_compiler_session_t"

  let error : error structure typ = structure "iree_compiler_error_t"

  let source : source structure typ = structure "iree_compiler_source_t"

  let output : output structure typ = structure "iree_compiler_output_t"

  let invocation : invocation structure typ =
    structure "iree_compiler_invocation_t"

  let iree_compiler_pipeline_std = constant "IREE_COMPILER_PIPELINE_STD" int64_t

  let iree_compiler_pipeline_hal_executable =
    constant "IREE_COMPILER_PIPELINE_HAL_EXECUTABLE" int64_t

  let iree_compiler_pipeline_precompile =
    constant "IREE_COMPILER_PIPELINE_PRECOMPILE" int64_t

  let pipeline =
    enum "iree_compiler_pipeline_t"
      [ (Std, iree_compiler_pipeline_std)
      ; (Hal_executable, iree_compiler_pipeline_hal_executable)
      ; (Precompile, iree_compiler_pipeline_precompile) ]
end
