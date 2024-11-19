import torch, time
gpu = 0
b, c, s = 100, 200, 100
# x = torch.rand(size=(b, c, s, s)).cuda(gpu)
xx1 = torch.rand(size=(b, 1, 1, 1)).cuda(gpu)
xx2 = torch.rand(size=(b, c, s, 1)).cuda(gpu)
xx3 = torch.rand(size=(b, 1, s, s)).cuda(gpu)
xx4 = torch.rand(size=(b, c, s, s)).cuda(gpu)

def calculate(f=lambda x:x-xx1, name='subtrc', range=(0, 1), default=None):
    input = torch.rand(size=(b, c, s, s)).cuda(gpu) * (range[1]-range[0]) + range[0]
    start = time.time()
    output = f(input)
    elapsed = time.time()-start

    if default is None:
        default = elapsed

    print(f"{name:6}  {elapsed:.6f}  {elapsed/default:.4e} times", end='')
    print(f"  {'big' if elapsed/default > 10 else 'ignore'}")
    return output, elapsed

_, default = calculate() # subtract - 1 FLOP/element
_, elapsed = calculate(lambda input: input-xx4, 'subtr4', (-1, 1), default) # subtract - 1 FLOP/element
_, elapsed = calculate(lambda input: input-xx1, 'add1', (-1, 1), default) # subtract - 1 FLOP/element
_, elapsed = calculate(lambda input: input-xx4, 'add4', (-1, 1), default) # subtract - 1 FLOP/element
_, elapsed = calculate(lambda input: input*xx1, 'mulou1', (-1, 1), default) # mul - 1 FLOP/element
_, elapsed = calculate(lambda input: input*xx2, 'mulou2', (-1, 1), default) # mul - 1 FLOP/element
_, elapsed = calculate(lambda input: input*xx3, 'mulou3', (-1, 1), default) # mul - 1 FLOP/element
_, elapsed = calculate(lambda input: input*xx4, 'mulou4', (-1, 1), default) # mul - 1 FLOP/element
_, elapsed = calculate(lambda input: input.mul_(xx1), 'mulin1', (-1, 1), default) # mul_ (inplace) - 1 FLOP/element
_, elapsed = calculate(lambda input: input.mul_(xx2), 'mulin2', (-1, 1), default) # mul_ (inplace) - 1 FLOP/element
_, elapsed = calculate(lambda input: input.mul_(xx3), 'mulin3', (-1, 1), default) # mul_ (inplace) - 1 FLOP/element
_, elapsed = calculate(lambda input: input.mul_(xx4), 'mulin4', (-1, 1), default) # mul_ (inplace) - 1 FLOP/element
_, elapsed = calculate(lambda input: torch.tanh(input), 'tanh', (-8, 8), default) # tanh - ignorable
_, elapsed = calculate(lambda input: torch.acos(input), 'acos', (-1, 1), default) # acos - 0.5 FLOP/element (but ignorable)
_, elapsed = calculate(lambda input: torch.cos(input), 'cosine', (-3.14, 3.14), default) # cos - ignorable
_, elapsed = calculate(lambda input: input.unsqueeze(dim=1).unsqueeze(dim=1).squeeze(), 'squeez', (-1, 1), default) # squeeze & unsqueeze - ignorable
_, elapsed = calculate(lambda input: torch.clamp(input, -1+1e-7, 1-1e-7), 'clamp', (-1, 1), default) # clamp - 1~1.5 FLOP/element
_, elapsed = calculate(lambda input: input.tile(1,1,1,4), 'tile', (-1, 1), default) # tile - 1 FLOP/element
_, elapsed = calculate(lambda input: input.repeat(1,1,1,4), 'repeat', (-1, 1), default) # repeat - 1 FLOP/element
_, elapsed = calculate(lambda input: input.mean(), 'mean', (-1, 1), default) # mean - 1~2 FLOPs/element
