open Ctypes

module Types (F : TYPE) = struct
  open F

  type client

  type error

  type api

  type loaded_executable

  type device

  type event

  type buffer

  type buffer_type = F32 | F64 | I1 | I64 | U32 | U64

  type host_buffer_semantics =
    | ImmutableOnlyDuringCall
    | ImmutableUntilTransferCompletes
    | ImmutableZeroCopy
    | MutableZeroCopy

  type memory

  type buffer_memory_layout

  type executable

  type serialized_executable

  let client : client structure typ = structure "PJRT_Client"

  let error : error structure typ = structure "PJRT_Error"

  let api : api structure typ = structure "PJRT_Api"

  module ErrorDestroy = struct
    type input = error structure ptr

    type output = unit

    type t

    type status = unit

    let t : t structure typ = structure "PJRT_Error_Destroy_Args"

    let status = void

    let struct_size = field t "struct_size" size_t

    let error = field t "error" (ptr error)

    let () = seal t

    let api_field =
      field api "PJRT_Error_Destroy"
      @@ static_funptr (ptr t @-> returning status)

    let check_status _ () = ()
  end

  module ErrorMessage = struct
    type input = error structure ptr

    type output = string

    type t

    type status = unit

    let t : t structure typ = structure "PJRT_Error_Message_Args"

    let status = void

    let struct_size = field t "struct_size" size_t

    let error = field t "error" (const @@ ptr error)

    let message = field t "message" string

    let () = seal t

    let offset = 48

    let api_field =
      field api "PJRT_Error_Message"
      @@ static_funptr (ptr t @-> returning status)

    let check_status _ () = ()
  end

  module PluginInitialize = struct
    type input = unit

    type output = unit

    type t

    let t : t structure typ = structure "PJRT_Plugin_Initialize_Args"

    let struct_size = field t "struct_size" size_t

    let () = seal t

    let api_field =
      field api "PJRT_Plugin_Initialize"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))

    let of_input () _ = ()

    let to_output _ = ()
  end

  module Destroy (F : sig
    val name : string

    val field_name : string

    type t

    val t : t structure typ
  end) =
  struct
    type t' = F.t

    type t

    let t : t structure typ = structure (F.name ^ "_Destroy_Args")

    let struct_size = field t "struct_size" size_t

    let element = field t F.field_name (ptr F.t)

    let () = seal t

    let api_field =
      field api (F.name ^ "_Destroy")
      @@ static_funptr (ptr t @-> returning (ptr_opt error))
  end

  module ClientCreate = struct
    type input = unit

    type output = client structure ptr

    type t

    let t : t structure typ = structure "PJRT_Client_Create_Args"

    let struct_size = field t "struct_size" size_t

    let client = field t "client" (ptr client)

    let () = seal t

    let api_field =
      field api "PJRT_Client_Create"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))

    let of_input () _ = ()
  end

  module ClientDestroy = Destroy (struct
    let name = "PJRT_Client"

    let field_name = "client"

    type t = client

    let t = client
  end)

  module Program = struct
    type t

    let t : t structure typ = structure "PJRT_Program"

    let struct_size = field t "struct_size" size_t

    let code = field t "code" string

    let code_size = field t "code_size" size_t

    let format = field t "format" string

    let format_size = field t "format_size" size_t

    let () = seal t
  end

  let loaded_executable : loaded_executable structure typ =
    structure "PJRT_LoadedExecutable"

  module ClientCompile = struct
    type input = client structure ptr * Program.t structure ptr

    type output = loaded_executable structure ptr

    type t

    let t : t structure typ = structure "PJRT_Client_Compile_Args"

    let struct_size = field t "struct_size" size_t

    let client = field t "client" (ptr client)

    let program = field t "program" (ptr Program.t)

    let compile_options = field t "compile_options" string

    let compile_options_size = field t "compile_options_size" size_t

    let executable = field t "executable" (ptr loaded_executable)

    let () = seal t

    let api_field =
      field api "PJRT_Client_Compile"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))
  end

  let device : device structure typ = structure "PJRT_Device"

  module ClientDevices = struct
    type input = client structure ptr

    type output = device structure ptr list

    type t

    let t : t structure typ = structure "PJRT_Client_Devices_Args"

    let struct_size = field t "struct_size" size_t

    let client = field t "client" (ptr client)

    let devices = field t "devices" (ptr @@ ptr device)

    let num_devices = field t "num_devices" size_t

    let () = seal t

    let api_field =
      field api "PJRT_Client_Devices"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))
  end

  let event : event structure typ = structure "PJRT_Event"

  module EventAwait = struct
    type input = event structure ptr

    type output = unit

    type t

    let t : t structure typ = structure "PJRT_Event_Await_Args"

    let struct_size = field t "struct_size" size_t

    let event = field t "event" (ptr event)

    let () = seal t

    let api_field =
      field api "PJRT_Event_Await"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))
  end

  module EventDestroy = Destroy (struct
    let name = "PJRT_Event"

    let field_name = "event"

    type t = event

    let t = event
  end)

  let buffer : buffer structure typ = structure "PJRT_Buffer"

  let buffer_type =
    enum ~typedef:true "PJRT_Buffer_Type"
      [ (F32, constant "PJRT_Buffer_Type_F32" int64_t)
      ; (F64, constant "PJRT_Buffer_Type_F64" int64_t)
      ; (I1, constant "PJRT_Buffer_Type_PRED" int64_t)
      ; (I64, constant "PJRT_Buffer_Type_S64" int64_t)
      ; (U32, constant "PJRT_Buffer_Type_U32" int64_t)
      ; (U64, constant "PJRT_Buffer_Type_U64" int64_t) ]

  let host_buffer_semantics =
    enum ~typedef:true "PJRT_HostBufferSemantics"
      [ ( ImmutableOnlyDuringCall
        , constant "PJRT_HostBufferSemantics_kImmutableOnlyDuringCall" int64_t
        )
      ; ( ImmutableUntilTransferCompletes
        , constant "PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes"
            int64_t )
      ; ( ImmutableZeroCopy
        , constant "PJRT_HostBufferSemantics_kImmutableZeroCopy" int64_t )
      ; ( MutableZeroCopy
        , constant "PJRT_HostBufferSemantics_kMutableZeroCopy" int64_t ) ]

  let memory : memory structure typ = structure "PJRT_Memory"

  let buffer_memory_layout : buffer_memory_layout structure typ =
    structure "PJRT_Buffer_MemoryLayout"

  module BufferFromHostBuffer = struct
    type input =
      | Input :
          client structure ptr
          * device structure ptr
          * ('a, 'b) Device_api.Tensor.t
          -> input

    type output = buffer structure ptr * event structure ptr

    type t

    let t : t structure typ = structure "PJRT_Client_BufferFromHostBuffer_Args"

    let struct_size = field t "struct_size" size_t

    let client = field t "client" (ptr client)

    let data = field t "data" (ptr void)

    let type' = field t "type" buffer_type

    let dims = field t "dims" (ptr int64_t)

    let num_dims = field t "num_dims" size_t

    let host_buffer_semantics =
      field t "host_buffer_semantics" host_buffer_semantics

    let device = field t "device" (ptr device)

    let event = field t "done_with_host_buffer" (ptr event)

    let buffer = field t "buffer" (ptr buffer)

    let () = seal t

    let api_field =
      field api "PJRT_Client_BufferFromHostBuffer"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))
  end

  module ExecuteOptions = struct
    type t

    let t : t structure typ = structure "PJRT_ExecuteOptions"

    let struct_size = field t "struct_size" size_t

    let non_donatable_input_indices =
      field t "non_donatable_input_indices" (ptr int64_t)

    let num_non_donatable_input_indices =
      field t "num_non_donatable_input_indices" size_t

    let () = seal t
  end

  module LoadedExecutableExecute = struct
    type input =
      loaded_executable structure ptr
      * ExecuteOptions.t structure
      * buffer structure ptr list
      * buffer structure ptr ptr

    type output = event structure ptr

    type t

    let t : t structure typ = structure "PJRT_LoadedExecutable_Execute_Args"

    let struct_size = field t "struct_size" size_t

    let executable = field t "executable" (ptr loaded_executable)

    let options = field t "options" (ptr ExecuteOptions.t)

    let num_devices = field t "num_devices" size_t

    let num_args = field t "num_args" size_t

    let argument_lists = field t "argument_lists" (ptr @@ ptr @@ ptr buffer)

    let output_lists = field t "output_lists" (ptr @@ ptr @@ ptr buffer)

    let device_complete_events =
      field t "device_complete_events" (ptr @@ ptr event)

    let () = seal t

    let api_field =
      field api "PJRT_LoadedExecutable_Execute"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))
  end

  module LoadedExecutableDestroy = Destroy (struct
    let name = "PJRT_LoadedExecutable"

    let field_name = "executable"

    type t = loaded_executable

    let t = loaded_executable
  end)

  module BufferToHostBuffer = struct
    type input = buffer structure ptr * int

    type output = unit ptr * event structure ptr

    type t

    let t : t structure typ = structure "PJRT_Buffer_ToHostBuffer_Args"

    let struct_size = field t "struct_size" size_t

    let src = field t "src" (ptr buffer)

    let dst = field t "dst" (ptr void)

    let dst_size = field t "dst_size" size_t

    let event = field t "event" (ptr event)

    let () = seal t

    let api_field =
      field api "PJRT_Buffer_ToHostBuffer"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))
  end

  module BufferDestroy = Destroy (struct
    let name = "PJRT_Buffer"

    let field_name = "buffer"

    type t = buffer

    let t = buffer
  end)

  let executable : executable structure typ = structure "PJRT_Executable"

  module LoadedExecutableGetExecutable = struct
    type input = loaded_executable structure ptr

    type output = executable structure ptr

    type t

    let t : t structure typ =
      structure "PJRT_LoadedExecutable_GetExecutable_Args"

    let struct_size = field t "struct_size" size_t

    let loaded_executable = field t "loaded_executable" (ptr loaded_executable)

    let executable = field t "executable" (ptr executable)

    let () = seal t

    let api_field =
      field api "PJRT_LoadedExecutable_GetExecutable"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))
  end

  let serialized_executable : serialized_executable structure typ =
    structure "PJRT_SerializedExecutable"

  module ExecutableSerialize = struct
    type input = executable structure ptr

    type output = string

    type t

    let t : t structure typ = structure "PJRT_Executable_Serialize_Args"

    let struct_size = field t "struct_size" size_t

    let executable = field t "executable" (ptr executable)

    let serialized_bytes = field t "serialized_bytes" (ptr char)

    let serialized_bytes_size = field t "serialized_bytes_size" size_t

    let serialized_executable_deleter =
      field t "serialized_executable_deleter"
        (static_funptr (ptr serialized_executable @-> returning void))

    let serialized_executable' =
      field t "serialized_executable" (ptr serialized_executable)

    let () = seal t

    let api_field =
      field api "PJRT_Executable_Serialize"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))
  end

  module ExecutableDeserializeAndLoad = struct
    type input = client structure ptr * string

    type output = loaded_executable structure ptr

    type t

    let t : t structure typ =
      structure "PJRT_Executable_DeserializeAndLoad_Args"

    let struct_size = field t "struct_size" size_t

    let client = field t "client" (ptr client)

    let serialized_executable = field t "serialized_executable" string

    let serialized_executable_size = field t "serialized_executable_size" size_t

    let loaded_executable = field t "loaded_executable" (ptr loaded_executable)

    let () = seal t

    let api_field =
      field api "PJRT_Executable_DeserializeAndLoad"
      @@ static_funptr (ptr t @-> returning (ptr_opt error))
  end

  module ExecutableDestroy = Destroy (struct
    let name = "PJRT_Executable"

    let field_name = "executable"

    type t = executable

    let t = executable
  end)

  let () = seal api
end
