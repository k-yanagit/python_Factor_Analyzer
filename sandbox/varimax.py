import numpy as np

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = np.eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u,s,vh = np.linalg.svd(np.dot(Phi.T,np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda))))))
        R = np.dot(u,vh)
        d = np.sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return np.dot(Phi, R)

# 例としてランダムなデータを生成
np.random.seed(0)
data = np.random.rand(100, 5)

# 仮に因子負荷量行列をランダムに生成（実際には因子分析の結果を使う）
loadings = np.random.rand(5, 2)

# バリマックス回転を実行
rotated_loadings = varimax(loadings)
print(rotated_loadings)