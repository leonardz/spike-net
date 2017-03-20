from tensorflow.python.framework import ops

@ops.RegisterGradient("RandomPoisson")
def _poisson_grad(op, grads):
  return None, grads
