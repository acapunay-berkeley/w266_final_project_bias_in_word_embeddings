import numpy as np
import pandas as pd

def neutralize(u, v):
    return u - v * u.dot(v) / v.dot(v)

def debias(embedding, bias_direction, equalize):
    
    print("Neutralizing")
    for i, _ in enumerate(embedding.index):
        if i % 100000 == 0:
            print(i)
        embedding.iloc[i] = neutralize(np.array(embedding.iloc[i]), bias_direction)
    
    # Normalize
    embedding = pd.DataFrame(embedding.to_numpy() / np.linalg.norm(embedding.to_numpy(), axis=1)[:, np.newaxis], index=embedding.index, dtype='f')
    
    for (a, b) in equalize:
        va = np.array(embedding[embedding.index == a])[0]
        vb = np.array(embedding[embedding.index == b])[0]
        y = neutralize((va + vb) / 2, bias_direction)
        z = np.sqrt(1 - np.linalg.norm(y)**2)
        if (va + vb).dot(bias_direction) < 0:
            z = -z
        embedding[embedding.index == a] = (z * bias_direction + y).reshape(1, -1)
        embedding[embedding.index == b] = (-z * bias_direction + y).reshape(1, -1)
        
    # Normalize one more time
    embedding = pd.DataFrame(embedding.to_numpy() / np.linalg.norm(embedding.to_numpy(), axis=1)[:, np.newaxis], index=embedding.index, dtype='f')
    
    return embedding