module type S = sig
  type 'a u

  type !'a v

  type _ t =
    | [] : ('a * 'a) t
    | ( :: ) : 'a u * ('b * 'c) t -> ('b * ('a v -> 'c)) t

  type 'a fold_fn = {f: 'b. 'a -> 'b u -> 'a}

  val fold_left : 'a fold_fn -> 'a -> 'b t -> 'a

  type 'a map_fn = {f: 'b. 'b u -> 'a}

  val map : 'a map_fn -> 'b t -> 'a list

  type any = Any : 'a u -> any

  val to_any_list : 'a t -> any list
end

module Make (T : sig
  type 'a t

  type !'a tag
end) : S with type 'a u = 'a T.t and type 'a v = 'a T.tag = struct
  type 'a u = 'a T.t

  type 'a v = 'a T.tag

  type _ t =
    | [] : ('a * 'a) t
    | ( :: ) : 'a T.t * ('b * 'c) t -> ('b * ('a v -> 'c)) t

  type 'a fold_fn = {f: 'b. 'a -> 'b T.t -> 'a}

  let rec fold_left : type a b. b fold_fn -> b -> a t -> b =
   fun f acc l ->
    match l with [] -> acc | x :: xs -> fold_left f (f.f acc x) xs

  type 'a map_fn = {f: 'b. 'b T.t -> 'a}

  let map f l = fold_left {f= (fun acc x -> List.cons (f.f x) acc)} [] l

  type any = Any : 'a T.t -> any

  let rec to_any_list : type a. a t -> any list = function
    | x :: xs ->
        Any x :: to_any_list xs
    | [] ->
        []
end
