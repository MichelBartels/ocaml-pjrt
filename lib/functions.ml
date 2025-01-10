open Ctypes
include Types_generated

module Function (F : sig
  type input

  type output

  type t

  type status

  val t : t structure typ

  val status : status typ

  val struct_size : (Unsigned.size_t, t structure) field

  val api_field :
    ((t structure ptr -> status) static_funptr, api structure) field

  val of_input : input -> t structure -> unit

  val to_output : t structure -> output

  val check_status : api structure ptr -> status -> unit
end) =
struct
  type input = F.input

  type output = F.output

  let call api input =
    let args = allocate_n F.t ~count:1 in
    args |-> F.struct_size <-@ Unsigned.Size_t.of_int @@ sizeof F.t ;
    F.of_input input !@args ;
    let f = !@(api |-> F.api_field) in
    let f_type = ptr F.t @-> returning F.status in
    let f = coerce (static_funptr f_type) (Foreign.funptr f_type) f in
    F.check_status api @@ f args ;
    F.to_output !@args
end

module ErrorDestroy = Function (struct
  include ErrorDestroy

  let of_input error' args = setf args error error'

  let to_output _ = ()
end)

module ErrorMessage = Function (struct
  include ErrorMessage

  let of_input error' args = setf args error error'

  let to_output args = getf args message
end)

module FunctionWithError (F : sig
  type input

  type output

  type t

  val t : t structure typ

  val struct_size : (Unsigned.size_t, t structure) field

  val api_field :
    ( (t structure ptr -> error structure ptr option) static_funptr
    , api structure )
    field

  val of_input : input -> t structure -> unit

  val to_output : t structure -> output
end) =
Function (struct
  type input = F.input

  type output = F.output

  type t = F.t

  type status = error structure ptr option

  let t = F.t

  let status = ptr_opt error

  let struct_size = F.struct_size

  let api_field = F.api_field

  let of_input = F.of_input

  let to_output = F.to_output

  let check_status api = function
    | None ->
        ()
    | Some error ->
        let message = ErrorMessage.call api error in
        ErrorDestroy.call api error ;
        failwith message
end)

module PluginInitialize = FunctionWithError (PluginInitialize)

module Destroy (F : sig
  type t'

  type t

  val t : t structure typ

  val struct_size : (Unsigned.size_t, t structure) field

  val element : (t' structure ptr, t structure) field

  val api_field :
    ( (t structure ptr -> error structure ptr option) static_funptr
    , api structure )
    field
end) =
FunctionWithError (struct
  include F

  type input = F.t' structure ptr

  type output = unit

  let of_input element' args = setf args element element'

  let to_output _ = ()
end)

module ClientCreate = FunctionWithError (struct
  include ClientCreate

  let to_output args = getf args client
end)

module Program = struct
  include Program

  let make code' =
    let program = make t in
    setf program struct_size @@ Unsigned.Size_t.of_int @@ sizeof t ;
    setf program code code' ;
    setf program code_size @@ Unsigned.Size_t.of_int @@ String.length code' ;
    let format' = "mlir" in
    setf program format format' ;
    setf program format_size @@ Unsigned.Size_t.of_int @@ String.length format' ;
    program
end

module ClientDestroy = Destroy (ClientDestroy)

module ClientCompile = FunctionWithError (struct
  include ClientCompile

  let of_input (client', program') args =
    setf args client client' ;
    setf args program program' ;
    let compile_options' =
      "\x1a\x0e \x01(\x01J\x08\x08\x01\x10\x01\x1a\x02\x08\x00"
    in
    setf args compile_options compile_options' ;
    setf args compile_options_size
    @@ Unsigned.Size_t.of_int
    @@ String.length compile_options'

  let to_output args = getf args executable
end)

module ClientDevices = FunctionWithError (struct
  include ClientDevices

  let of_input client' args = setf args client client'

  let to_output args =
    let num_devices' = Unsigned.Size_t.to_int @@ getf args num_devices in
    let devices' = getf args devices in
    let devices' = CArray.from_ptr devices' num_devices' in
    CArray.to_list devices'
end)

module EventAwait = FunctionWithError (struct
  include EventAwait

  let of_input event' args = setf args event event'

  let to_output _ = ()
end)

module EventDestroy = Destroy (EventDestroy)

module BufferFromHostBuffer = FunctionWithError (struct
  include BufferFromHostBuffer

  let of_input (Input (client', device', tensor)) args =
    setf args client client' ;
    let open Device_api in
    let data' = Tensor.data tensor in
    let data' = to_voidp data' in
    setf args data data' ;
    setf args type' F32 ;
    let dims' = Tensor.shape tensor in
    let dims' = List.map Signed.Int64.of_int dims' in
    setf args dims @@ CArray.start @@ CArray.of_list int64_t dims' ;
    setf args num_dims @@ Unsigned.Size_t.of_int @@ List.length dims' ;
    setf args host_buffer_semantics ImmutableUntilTransferCompletes ;
    setf args device device'

  let to_output args = (getf args buffer, getf args event)
end)

module ExecuteOptions = struct
  include ExecuteOptions

  let make non_donatable_input_indices' =
    let options = make t in
    setf options struct_size @@ Unsigned.Size_t.of_int @@ sizeof t ;
    setf options non_donatable_input_indices
    @@ CArray.start @@ CArray.of_list int64_t
    @@ List.map Int64.of_int non_donatable_input_indices' ;
    options
end

module LoadedExecutableExecute = FunctionWithError (struct
  include LoadedExecutableExecute

  let of_input (executable', options', buffers', output) args =
    setf args executable executable' ;
    setf args options (addr options') ;
    setf args num_devices @@ Unsigned.Size_t.of_int 1 ;
    setf args num_args @@ Unsigned.Size_t.of_int @@ List.length buffers' ;
    let buffers' = CArray.of_list (ptr buffer) buffers' in
    let buffers' = CArray.start buffers' in
    let buffers' = CArray.of_list (ptr @@ ptr buffer) [buffers'] in
    setf args argument_lists @@ CArray.start buffers' ;
    let output_lists' = CArray.of_list (ptr @@ ptr buffer) [output] in
    setf args output_lists @@ CArray.start output_lists' ;
    let device_complete_events' = allocate_n (ptr event) ~count:1 in
    setf args device_complete_events device_complete_events'

  let to_output args = !@(getf args device_complete_events)
end)

module LoadedExecutableDestroy = Destroy (LoadedExecutableDestroy)

module BufferToHostBuffer = FunctionWithError (struct
  include BufferToHostBuffer

  let of_input (buffer, num_elements) args =
    setf args src buffer ;
    setf args dst_size @@ Unsigned.Size_t.of_int @@ (num_elements * 4) ;
    let dst' = allocate_n float ~count:num_elements in
    setf args dst @@ to_voidp dst'

  let to_output args =
    let dst = getf args dst in
    let event = getf args event in
    (dst, event)
end)

module BufferDestroy = Destroy (BufferDestroy)

module LoadedExecutableGetExecutable = FunctionWithError (struct
  include LoadedExecutableGetExecutable

  let of_input loaded_executable' args =
    setf args loaded_executable loaded_executable'

  let to_output args = getf args executable
end)

module ExecutableSerialize = FunctionWithError (struct
  include ExecutableSerialize

  let of_input executable' args = setf args executable executable'

  let to_output args =
    let serialized_bytes = getf args serialized_bytes in
    let serialized_bytes_size = getf args serialized_bytes_size in
    let string =
      string_from_ptr serialized_bytes
        ~length:(Unsigned.Size_t.to_int serialized_bytes_size)
    in
    let destructor = getf args serialized_executable_deleter in
    let destructor =
      coerce
        (static_funptr (ptr serialized_executable @-> returning void))
        (Foreign.funptr (ptr serialized_executable @-> returning void))
        destructor
    in
    let serialized_executable = getf args serialized_executable' in
    destructor serialized_executable ;
    string
end)

module ExecutableDeserializeAndLoad = FunctionWithError (struct
  include ExecutableDeserializeAndLoad

  let of_input (client', serialized_executable') args =
    setf args client client' ;
    setf args serialized_executable serialized_executable' ;
    setf args serialized_executable_size
    @@ Unsigned.Size_t.of_int
    @@ String.length serialized_executable'

  let to_output args = getf args loaded_executable
end)

module ExecutableDestroy = Destroy (ExecutableDestroy)
