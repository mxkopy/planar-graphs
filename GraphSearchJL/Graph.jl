struct Node
    x::Number
    y::Number
    parent::Node
    size::Int
end

struct Edge
    u::Node
    v::Node
end
