def register_poisson_gradient(tf, ops):
    @ops.RegisterGradient("RandomPoisson")
    def _poisson_grad(op, grads):
        return None, grads
