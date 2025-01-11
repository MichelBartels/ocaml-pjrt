open Ctypes
open Functions

type program = loaded_executable structure ptr

let () = at_exit Gc.full_major

type t = {api: api structure ptr; client: client structure ptr}

let make path =
  let lib = Dl.dlopen ?filename:(Some path) ~flags:[Dl.RTLD_LAZY] in
  let api_func =
    Foreign.foreign ?check_errno:(Some true) ~from:lib "GetPjrtApi"
      (void @-> returning (ptr api))
  in
  let api = api_func () in
  PluginInitialize.call api () ;
  let client = ClientCreate.call api () in
  let t = {api; client} in
  Gc.finalise
    (fun t ->
      print_endline "destroying client" ;
      ClientDestroy.call t.api t.client )
    t ;
  t

let compile t code =
  let program = Program.make code in
  let loaded_executable = ClientCompile.call t.api (t.client, addr program) in
  Gc.finalise (fun p -> LoadedExecutableDestroy.call t.api p) loaded_executable ;
  loaded_executable

let write t loaded_executable path =
  let executable = LoadedExecutableGetExecutable.call t.api loaded_executable in
  let serialized = ExecutableSerialize.call t.api executable in
  ExecutableDestroy.call t.api executable ;
  let ch = open_out_bin path in
  output_string ch serialized ;
  close_out ch

let read t path =
  let ch = open_in_bin path in
  let serialized = really_input_string ch (in_channel_length ch) in
  close_in ch ;
  let loaded_executable =
    ExecutableDeserializeAndLoad.call t.api (t.client, serialized)
  in
  Gc.finalise (fun p -> LoadedExecutableDestroy.call t.api p) loaded_executable ;
  loaded_executable

let devices t = ClientDevices.call t.api t.client

type buffer = {buffer: Types_generated.buffer structure ptr}

let finalise_buffer t buffer = BufferDestroy.call t.api buffer.buffer

let buffer_to_device t device tensor =
  let buffer, event =
    BufferFromHostBuffer.call t.api (Input (t.client, device, tensor))
  in
  EventAwait.call t.api event ;
  EventDestroy.call t.api event ;
  let buffer = {buffer} in
  buffer

let execute t num_outputs executable buffers =
  let internal_buffers = List.map (fun b -> b.buffer) buffers in
  let non_donatable = List.init (List.length buffers) Fun.id in
  (* let non_donatable = [] in *)
  let options = ExecuteOptions.make non_donatable in
  let output = allocate_n (ptr Types_generated.buffer) ~count:num_outputs in
  let event =
    LoadedExecutableExecute.call t.api
      (executable, options, internal_buffers, output)
  in
  EventAwait.call t.api event ;
  EventDestroy.call t.api event ;
  let buffers = CArray.to_list @@ CArray.from_ptr output num_outputs in
  let buffers = List.map (fun buffer -> {buffer}) buffers in
  buffers

let buffer_to_host t ctype num_elements buffer =
  let data, event = BufferToHostBuffer.call t.api (buffer, num_elements) in
  EventAwait.call t.api event ;
  EventDestroy.call t.api event ;
  let data = coerce (ptr void) (ptr ctype) data in
  CArray.from_ptr data num_elements
