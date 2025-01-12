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
  Gc.finalise (fun t -> ClientDestroy.call t.api t.client) t ;
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

let buffer_to_device : type a b.
    t -> device structure ptr -> (a, b) Device_api.Tensor.t -> buffer =
 fun t device tensor ->
  let open Device_api in
  let data = Tensor.data tensor in
  let kind = Tensor.kind tensor in
  let data = to_voidp data in
  let root_1 = Root.create data in
  let buffer_type =
    match kind with
    | F32 ->
        F32
    | F64 ->
        F64
    | I1 ->
        I1
    | I64 ->
        I64
    | U32 ->
        U32
    | U64 ->
        U64
  in
  let dims = Tensor.shape tensor in
  let num_dims = List.length dims in
  let dims = List.map Signed.Int64.of_int dims in
  let dims = CArray.of_list int64_t dims in
  let root_2 = Root.create dims in
  let buffer, event =
    BufferFromHostBuffer.call t.api
      (t.client, data, buffer_type, CArray.start dims, device, num_dims)
  in
  EventAwait.call t.api event ;
  EventDestroy.call t.api event ;
  Root.release root_1 ;
  Root.release root_2 ;
  {buffer}

let execute t num_outputs executable buffers =
  let internal_buffers = List.map (fun b -> b.buffer) buffers in
  let internal_buffers =
    CArray.of_list (ptr Types_generated.buffer) internal_buffers
  in
  let root_1 = Root.create internal_buffers in
  let internal_buffers = CArray.start internal_buffers in
  let internal_buffers =
    CArray.of_list (ptr @@ ptr Types_generated.buffer) [internal_buffers]
  in
  let root_2 = Root.create internal_buffers in
  let non_donatable = List.init (List.length buffers) Fun.id in
  (* let non_donatable = [] in *)
  let options = ExecuteOptions.make non_donatable in
  let root_3 = Root.create options in
  let output = allocate_n (ptr Types_generated.buffer) ~count:num_outputs in
  let output' = CArray.of_list (ptr @@ ptr buffer) [output] in
  let root_4 = Root.create output' in
  print_endline "before execute" ;
  let event = allocate_n (ptr Types_generated.event) ~count:1 in
  LoadedExecutableExecute.call t.api
    ( executable
    , options
    , CArray.start internal_buffers
    , List.length buffers
    , CArray.start output'
    , event ) ;
  print_endline "after execute" ;
  let event = !@event in
  EventAwait.call t.api event ;
  print_endline "after await" ;
  EventDestroy.call t.api event ;
  print_endline "after destroy" ;
  Root.release root_1 ;
  Root.release root_2 ;
  Root.release root_3 ;
  Root.release root_4 ;
  let buffers = CArray.to_list @@ CArray.from_ptr output num_outputs in
  let buffers = List.map (fun buffer -> {buffer}) buffers in
  buffers

let buffer_to_host t ctype num_elements buffer =
  let dst = allocate_n ctype ~count:num_elements in
  let dst = coerce (ptr ctype) (ptr void) dst in
  let dst_size = num_elements * sizeof ctype in
  let event = BufferToHostBuffer.call t.api (buffer, dst, dst_size) in
  EventAwait.call t.api event ;
  EventDestroy.call t.api event ;
  let data = coerce (ptr void) (ptr ctype) dst in
  CArray.from_ptr data num_elements
