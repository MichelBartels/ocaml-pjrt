module Make (M : sig
  val path : string
end) : Device_api.S = struct
  let client = Client.make M.path

  let device = Client.devices client |> List.hd

  type program = Client.program

  type buffer = Client.buffer

  let compile_and_store ~program ~path =
    let executable = Client.compile client program in
    Client.write client executable path ;
    executable

  let load ~path = Client.read client path

  let tensor_to_buffer tensor = Client.buffer_to_device client device tensor

  let buffer_to_tensor ~shape kind buffer =
    let open Device_api in
    let ctype = Tensor.ctype_of_kind kind in
    let num_elements = List.fold_left ( * ) 1 shape in
    let data =
      Client.buffer_to_host client ctype num_elements buffer.Client.buffer
    in
    Tensor.make kind shape data

  let execute program ~num_outputs inputs =
    Client.execute client num_outputs program inputs
end

let make path =
  ( module Make (struct
    let path = path
  end) : Device_api.S )
