let fn = Ir.create_func

module Var = Ir.Var

let ( + ) a b = Ir.tag (Var.Add (a, b))

let ( - ) a b = Ir.tag (Var.Subtract (a, b))

let ( * ) a b = Ir.tag (Var.Multiply (a, b))

let abs a = Ir.tag (Var.Abs a)

let compare dir a b = Ir.tag (Var.Compare (a, dir, b))

let ( = ) a = compare Ir.Eq a

let ( <> ) a = compare Ir.Ne a

let ( >= ) a = compare Ir.Ge a

let ( > ) a = compare Ir.Gt a

let ( <= ) a = compare Ir.Le a

let ( < ) a = compare Ir.Lt a

let full value shape = Ir.tag (Tensor.full value shape |> Tensor.to_ir)

let full_f32 value = full (F32 value)

let full_i1 value = full (I1 value)

let norm mean stddev shape =
  Ir.tag
    (Var.Random
       ( Ir.Tensor_type (shape, F32)
       , mean
       , stddev
       , Tensor.from_int_list shape |> Tensor.to_ir |> Ir.tag
       , Normal ) )

let uniform low high shape =
  Ir.tag
    (Var.Random
       ( Ir.Tensor_type (shape, F32)
       , low
       , high
       , Tensor.from_int_list shape |> Tensor.to_ir |> Ir.tag
       , Uniform ) )
