import numpy as np
def read_txt(path):
    f = open(path)
    data = [] 
    run = True
    while(run):
        row = f.readline()
        if row == '':
            break
        data.append(row)
    return data

data = read_txt('D:\\FRI\\Diploma\\data\\artificial_data_1.txt')
for ix, row in enumerate(data):
    data[ix] = data[ix].replace(',', '').replace('\n', '').split(' ')
X = np.array(data)[:,:2].astype(np.float64)
#X[:,0] *= 5
#X[:,1] *= 10000
#X[:,1] = np.where(X[:,0] < 17, 0, X[:,1])
y = np.array(data)[:,2].astype(np.int64)
y = np.where(y == 2, 0, y).astype(np.int64)

concatenated_matrix = np.hstack((X, np.expand_dims(y, axis=1)))

with open('D:\\FRI\\Diploma\\data\\artificial_data_1_new.txt', 'w') as f:
    for row in concatenated_matrix:
        line = ' '.join(str(element) for element in row) + '\n'
        f.write(line)