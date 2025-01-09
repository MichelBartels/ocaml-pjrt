open Ctypes
open Functions

type t = {api: api structure ptr; client: client structure ptr}

let make path =
  let lib = Dl.dlopen ?filename:(Some path) ~flags:[Dl.RTLD_LAZY] in
  print_endline "loading api" ;
  let api_func =
    Foreign.foreign ?check_errno:(Some true) ~from:lib "GetPjrtApi"
      (void @-> returning (ptr api))
  in
  print_endline @@ string_of_int @@ sizeof api ;
  print_endline "api loaded" ;
  let api = api_func () in
  print_endline "api created" ;
  PluginInitialize.call api () ;
  let client = ClientCreate.call api () in
  print_endline "client created" ;
  {api; client}

let compile t code =
  let program = Program.make code in
  ClientCompile.call t.api (t.client, addr program)

let devices t = ClientDevices.call t.api t.client

let buffer_to_device t device data shape =
  let buffer, event =
    BufferFromHostBuffer.call t.api (t.client, device, data, shape)
  in
  EventAwait.call t.api event ;
  buffer

let execute t num_outputs executable buffers =
  let options = ExecuteOptions.make () in
  let buffers, event =
    LoadedExecutableExecute.call t.api
      (executable, options, buffers, num_outputs)
  in
  EventAwait.call t.api event ;
  buffers

let buffer_to_host t num_elements buffer =
  print_endline "buffer_to_host" ;
  let data, event = BufferToHostBuffer.call t.api (buffer, num_elements * 3) in
  print_endline "buffer_to_host 2" ;
  EventAwait.call t.api event ;
  print_endline "buffer_to_host 3" ;
  let data = coerce (ptr void) (ptr float) data in
  let data = CArray.from_ptr data num_elements in
  CArray.to_list data
