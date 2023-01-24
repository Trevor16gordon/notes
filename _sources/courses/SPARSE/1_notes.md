


# All Notes



## Notes Mar 28
Contraints
- One example from the homework with a (epsilon) < param <= (1 - epsilon)
- You could just add your constraint into the cost
    - Why wouldn't you want to do that? ... As your multiple for that part of the cost increases
        - (Lipschitz constaint with gradient is really big)
        - Convergence becomes very slow
- Continuation Penality
    - start with the parameter for relative cost low
    - Solve, then use that point as the starting point for next iteration with a higher cost.
- Langrangian Duality
    - Primal is the original objective 
    - Langragian of x and "new" is g(x) + <new, AX-y>
    - D(new) is a lower bound 
    - Dual ascent
    - Assumptions:
        - Assume the optimal ppoint exists and is unique
        - d is differentiable (d = inf g(x))
    - Problem with Langrangian is you can't guaruntee that it's dual feasible (There is min)


- Augmented Langrangian
    - Additionally adds L2 cost ||Ax - y||2,2
    - lagrangian constants
    - Augmented Langrangian is like a proximal gradient type solution


- Variation on Augmented Langrangian Multiplier
    