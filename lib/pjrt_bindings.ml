module Make (M : sig
  val path : string
  val caching : bool
end) : Device_api.S = struct
  let client = Client.make M.path

  let device = Client.devices client |> List.hd

  type program = Client.program

  type buffer = Client.buffer

  let compile ?path program =
    if M.caching then
      match path with
      | Some path ->
          if Sys.file_exists path then
            Client.read client path
          else
            let executable = Client.compile client program in
            Client.write client executable path;
            executable
      | None -> Client.compile client program
    else
      Client.compile client program

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

  let identifier = "pjrt_" ^ M.path

  let collect_buffer = Client.finalise_buffer client
end

let make ?(caching = true) path =
  ( module Make (struct
    let path = path
    let caching = caching
  end) : Device_api.S )
