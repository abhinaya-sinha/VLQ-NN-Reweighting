import numpy as np
def LRP(model, rho, incr, X, Y):

    W = []
    B = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            W.append(param.cpu().detach().numpy().transpose())
        elif 'bias' in name:
            B.append(param.cpu().detach().numpy())
    L = len(W)

    A = [X]+[None]*L
    for l in range(L-1):
        A[l+1] = np.maximum(0.01*(A[l].dot(W[l])+B[l]),A[l].dot(W[l])+B[l]) #forward pass with LeakyRelu activation
    A[L] = A[L-1].dot(W[L-1])+B[L-1]

    R = [None]*L + [A[L]*Y[:, None]]

    for l in range(0,L)[::-1]:
        w = rho(W[l],l)
        b = rho(B[l],l)
        
        z = incr(A[l].dot(w)+b,l)                # step 1
        s = R[l+1] / z                           # step 2
        c = w.dot(s.transpose()).transpose()     # step 3
        R[l] = A[l]*c                            # step 4

    del A, W, B
    return R