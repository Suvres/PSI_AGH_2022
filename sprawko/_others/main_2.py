outputs = int(sys.argv[2])
file = sys.argv[1]
input_file = open(file)
weights = json.load(input_file)
weightsList = [len(w) for w in weights]
n = net(weightsList, outputs)
while True:
    com = input()
    args = np.array(com.split(' '))
    if args[0] == '0':
        exit(0)
    args = args[1:].astype(np.float64)
    outputs = n.calc(weights=weights, inputs=args)
    sys.stdout.write('{0:f}'.format(outputs[0]))
    for i in range(1, len(outputs)):
        sys.stdout.write(' {0:f}'.format(outputs[i]))    
    sys.stdout.write('\n')
    sys.stdout.flush()
