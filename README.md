# Non-Linear Vector Auto Regression with Missing Data
Topology identification from multiple time series has been proved to be useful for system identification, anomaly detection, denoising, and data completion. Vector autoregressive (VAR) methods have proved well in identifying directed topology from complex networks. The task of inferring topology in the presence of noise and missing observations has been studied for linear models. As a first approach to joint signal estimation and topology identification with a nonlinear model, this paper proposes a method to do so under the modelling assumption that signals are generated by a sparse VAR model in a latent space and then transformed by a set of invertible, component-wise nonlinearities. A non convex optimization problem is formed with lasso regularisation and solved via block coordinate descent (BCD). Initial experiments conducted on synthetic data sets show the identifying capability of the proposed method.% identifies the true topology and signal from noisy and missing observations


![Screenshot 2022-11-23 at 17 04 34](https://user-images.githubusercontent.com/64849646/203593587-28e558c4-37ea-41a2-a31f-1af011cd3c2e.png)

![Screenshot 2022-11-23 at 17 05 07](https://user-images.githubusercontent.com/64849646/203593832-7a3fcc1f-974e-442e-99a5-324906dfd5f9.png)


![Screenshot 2022-11-23 at 17 05 31](https://user-images.githubusercontent.com/64849646/203593847-288103d0-9d25-437b-bef2-ba0019b1dcef.png)
