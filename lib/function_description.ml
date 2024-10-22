open Ctypes
module Types = Types_generated

module Functions (F : FOREIGN) = struct
  open F

  let run_main =
    foreign "ireeCompilerRunMain" (int @-> ptr (ptr char) @-> returning int)

  let global_initialize =
    foreign "ireeCompilerGlobalInitialize" (void @-> returning void)

  let session_create =
    foreign "ireeCompilerSessionCreate"
      (void @-> returning (ptr_opt Types.session))

  let session_set_flags =
    foreign "ireeCompilerSessionSetFlags"
      (ptr Types.session @-> int @-> ptr string @-> returning void)

  let source_wrap_buffer =
    foreign "ireeCompilerSourceWrapBuffer"
      ( ptr Types.session @-> string @-> string @-> size_t @-> bool
      @-> ptr (ptr Types.source)
      @-> returning (ptr_opt Types.error) )

  let error_get_message =
    foreign "ireeCompilerErrorGetMessage" (ptr Types.error @-> returning string)

  let error_destroy =
    foreign "ireeCompilerErrorDestroy" (ptr Types.error @-> returning void)

  let session_destroy =
    foreign "ireeCompilerSessionDestroy" (ptr Types.session @-> returning void)

  let output_open_file =
    foreign "ireeCompilerOutputOpenFile"
      (string @-> ptr (ptr Types.output) @-> returning (ptr_opt Types.error))

  let invocation_create =
    foreign "ireeCompilerInvocationCreate"
      (ptr Types.session @-> returning (ptr_opt Types.invocation))

  let invocation_parse_source =
    foreign "ireeCompilerInvocationParseSource"
      (ptr Types.invocation @-> ptr Types.source @-> returning bool)

  let invocation_destroy =
    foreign "ireeCompilerInvocationDestroy"
      (ptr Types.invocation @-> returning void)

  let invocation_pipeline =
    foreign "ireeCompilerInvocationPipeline"
      (ptr Types.invocation @-> Types.pipeline @-> returning bool)

  let invocation_output_vm_bytecode =
    foreign "ireeCompilerInvocationOutputVMBytecode"
      ( ptr Types.invocation @-> ptr Types.output
      @-> returning (ptr_opt Types_generated.error) )
end
