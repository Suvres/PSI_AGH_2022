def transpose(A):
    M = len(A)
    N = len(A[0])
    B = [[0 for x in range(M)] for y in range(N)]
    for i in range(N):
        for j in range(M):
            B[i][j] = A[j][i]
    return B

class net(nn.Module):
    def __init__(self, inputs: List, outputs):
        super(net, self).__init__()
        self.fc = []
        self.sig = nn.Sigmoid()
        for i in range(len(inputs)):
            i_size = inputs[i]
            o_size = inputs[i + 1] - 1 if i < len(inputs) - 1 else outputs
            
            self.fc = [*self.fc, nn.Linear(i_size, o_size, bias=False)]

    def calc(self, weights: List, inputs):
        output = inputs
        for i in range(len(weights)):
            output = [*output, 1.0]
            output = output
            layer = self.fc[i]
            with torch.no_grad():
                layer.weight = nn.Parameter(torch.Tensor(transpose(weights[i])))
            output = layer(torch.Tensor(output))
            if i < len(weights) - 1:
                output = self.sig(output)
        return output

