#
import time
import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .StPedf_module import SEDR_module, SEDR_impute_module
from tqdm import tqdm
# 在文件顶部添加以下依赖
import os
import anndata
import scanpy as sc
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix, coo_matrix

def target_distribution(batch):
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def reconstruction_loss(decoded, x):
    loss_func = torch.nn.MSELoss()
    loss_rcn = loss_func(decoded, x)
    return loss_rcn


# def gcn_loss(preds, labels, mu, logvar, n_nodes, norm, mask=None):
#     if mask is not None:
#         preds = preds * mask
#         labels = labels * mask
#
#     cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
#
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 / n_nodes * torch.mean(torch.sum(
#         1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
#     return cost + KLD



def gcn_loss(preds, labels, mu, logvar, n_nodes, norm):
    # preds: 形状 [n_edges]
    # labels: 形状 [n_edges]
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    return cost + KLD


class Sedr:
    def __init__(
            self,
            X,
            graph_dict,
            rec_w=10,
            gcn_w=0.1,
            self_w=1,
            dec_kl_w=1,
            mode = 'clustering',
            device = 'cpu',
    ):

        self.rec_w = rec_w
        self.gcn_w = gcn_w
        self.self_w = self_w
        self.dec_kl_w = dec_kl_w
        self.device = device
        self.mode = mode

        if 'mask' in graph_dict:
            self.mask = True
            self.adj_mask = graph_dict['mask'].to(self.device)
        else:
            self.mask = False

        self.cell_num = len(X)

        self.X = torch.FloatTensor(X.copy()).to(self.device)
        self.input_dim = self.X.shape[1]
        self.adj_norm = graph_dict["adj_norm"].to(self.device)
        self.adj_label = graph_dict["adj_label"].to(self.device)

        self.norm_value = graph_dict["norm_value"]

        if self.mode == 'clustering':
            self.model = SEDR_module(self.input_dim).to(self.device)
        elif self.mode == 'imputation':
            self.model = SEDR_impute_module(self.input_dim).to(self.device)
        else:
            raise ValueError(f'{self.mode} is not currently supported!')


    def mask_generator(self, N=1):
        idx = self.adj_label.indices()

        list_non_neighbor = []
        for i in range(0, self.cell_num):
            neighbor = idx[1, torch.where(idx[0, :] == i)[0]]
            n_selected = len(neighbor) * N

            # non neighbors
            total_idx = torch.range(0, self.cell_num-1, dtype=torch.float32).to(self.device)
            non_neighbor = total_idx[~torch.isin(total_idx, neighbor)]
            indices = torch.randperm(len(non_neighbor), dtype=torch.float32).to(self.device)
            random_non_neighbor = indices[:n_selected]
            list_non_neighbor.append(random_non_neighbor)

        x = torch.repeat_interleave(self.adj_label.indices()[0], N)
        y = torch.concat(list_non_neighbor)

        indices = torch.stack([x, y])
        indices = torch.concat([self.adj_label.indices(), indices], axis=1)

        value = torch.concat([self.adj_label.values(), torch.zeros(len(x), dtype=torch.float32).to(self.device)])
        adj_mask = torch.sparse_coo_tensor(indices, value)

        return adj_mask

    def train_without_dec(
            self,
            epochs=200,
            lr=0.01,
            decay=0.01,
            N=1,
    ):
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=lr,
            weight_decay=decay
        )

        self.model.train()

        for _ in tqdm(range(epochs)):
            self.model.train()
            self.optimizer.zero_grad()
            # 解包8个值（移除最后的占位符）
            latent_z, mu, logvar, de_feat, _, feat_x, _, loss_self = self.model(self.X, self.adj_norm)
            if self.mask:
                pass
            else:

                if self.mode == 'imputation':
                    adj_mask = self.mask_generator(N=0)
                else:
                    adj_mask = self.mask_generator(N=1)
                self.adj_mask = adj_mask
                self.mask = True

            # 计算 GCN 损失时传入边列表的预测值和标签
            preds = self.model.dc(latent_z, self.adj_mask)  # 形状 [n_edges]
            labels = self.adj_mask.coalesce().values().float()  # 形状 [n_edges]
            
            loss_gcn = gcn_loss(
                preds=preds,
                labels=labels,
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
            )

            loss_rec = reconstruction_loss(de_feat, self.X)
            loss = self.rec_w * loss_rec + self.gcn_w * loss_gcn + self.self_w * loss_self
            loss.backward()
            self.optimizer.step()

        #     list_rec.append(loss_rec.detach().cpu().numpy())
        #     list_gcn.append(loss_gcn.detach().cpu().numpy())
        #     list_self.append(loss_self.detach().cpu().numpy())
        #
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(list_rec, label='rec')
        # ax.plot(list_gcn, label='gcn')
        # ax.plot(list_self, label='self')
        # ax.legend()
        # plt.show()


    def save_model(self, save_model_file):
        torch.save({'state_dict': self.model.state_dict()}, save_model_file)
        print('Saving model to %s' % save_model_file)

    def load_model(self, save_model_file):
        saved_state_dict = torch.load(save_model_file)
        self.model.load_state_dict(saved_state_dict['state_dict'])
        print('Loading model from %s' % save_model_file)

    def process(self):
        self.model.eval()
        # 触发前向传播以生成属性（确保 latent_z 和 reconstructed_adj 被存储）
        with torch.no_grad():
            _ = self.model(self.X, self.adj_norm)  # 调用 forward 方法
        
        # 从类属性中获取数据
        latent_z = self.model.latent_z.detach().cpu().numpy()
        reconstructed_adj = self.model.reconstructed_adj.detach().cpu().numpy()
        
        # 其他处理（例如获取 q, feat_x, gnn_z）
        _, _, _, _, q, feat_x, gnn_z, _ = self.model(self.X, self.adj_norm)
        q = q.detach().cpu().numpy()
        feat_x = feat_x.detach().cpu().numpy()
        gnn_z = gnn_z.detach().cpu().numpy()
        
        # 将 latent_z 赋值给 Sedr 类的属性
        self.latent_z = latent_z
        
        # 返回所有必要数据（包括 reconstructed_adj）
        return latent_z, q, feat_x, gnn_z, reconstructed_adj
    def recon(self):
        self.model.eval()
        latent_z, _, _, de_feat, q, feat_x, gnn_z, _ = self.model(self.X, self.adj_norm)
        de_feat = de_feat.data.cpu().numpy()

        # revise std and mean
        from sklearn.preprocessing import StandardScaler
        out = StandardScaler().fit_transform(de_feat)

        return out

    def train_with_dec(
            self,
            epochs=200,
            dec_interval=20,
            dec_tol=0.00,
            N=1,
    ):
        self.train_without_dec()

        kmeans = KMeans(n_clusters=self.model.dec_cluster_n, n_init=self.model.dec_cluster_n * 2, random_state=42)
        # 解包5个返回值（latent_z, q, feat_x, gnn_z, reconstructed_adj）
        test_z, _, _, _, _ = self.process()
        y_pred_last = np.copy(kmeans.fit_predict(test_z))

        self.model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(self.device)
        self.model.train()

        for epoch_id in tqdm(range(epochs)):
            if epoch_id % dec_interval == 0:
                # 解包5个返回值
                _, tmp_q, _, _, _ = self.process()
                tmp_p = target_distribution(torch.Tensor(tmp_q))
                y_pred = tmp_p.cpu().numpy().argmax(1)
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                self.model.train()
                if epoch_id > 0 and delta_label < dec_tol:
                    print('delta_label {:.4}'.format(delta_label), '< tol', dec_tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            # 训练模型部分
            torch.set_grad_enabled(True)
            self.optimizer.zero_grad()
            # 解包8个值
            latent_z, mu, logvar, de_feat, out_q, _, _, loss_self = self.model(self.X, self.adj_norm)
            # if self.mask:
            #     pass
            # else:
            #     adj_mask = self.mask_generator(N)
            #     self.adj_mask = adj_mask
            #     self.mask = True

            loss_gcn = gcn_loss(
                preds=self.model.dc(latent_z, self.adj_mask),
                labels=self.adj_mask.coalesce().values(),
                mu=mu,
                logvar=logvar,
                n_nodes=self.cell_num,
                norm=self.norm_value,
                # mask=adj_mask,
            )
            loss_rec = reconstruction_loss(de_feat, self.X)
            # clustering KL loss
            loss_kl = F.kl_div(out_q.log(), torch.tensor(tmp_p).to(self.device)).to(self.device)
            loss = self.gcn_w * loss_gcn + self.dec_kl_w * loss_kl + self.rec_w * loss_rec
            loss.backward()
            self.optimizer.step()
    def pseudo_Spatiotemporal_Map(self, pSM_values_save_filepath="./pSM_values.tsv", n_neighbors=20, resolution=1.0):
        """
        Perform pseudo-Spatiotemporal Map for ST data
        :param pSM_values_save_filepath: the default save path for the pSM values
        :type pSM_values_save_filepath: class:`str`, optional, default: "./pSM_values.tsv"
        :param n_neighbors: The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.pp.neighbors.html` for detail
        :type n_neighbors: int, optional, default: 20
        :param resolution: A parameter value controlling the coarseness of the clustering.
        Higher values lead to more clusters. See `https://scanpy.readthedocs.io/en/stable/generated/scanpy.tl.leiden.html` for detail
        :type resolution: float, optional, default: 1.0
        """
        error_message = "No embedding found, please ensure you have run train() method before calculating pseudo-Spatiotemporal Map!"
        max_cell_for_subsampling = 5000
        try:
            print("Performing pseudo-Spatiotemporal Map")
            adata = anndata.AnnData(self.latent_z)
            
            # 检查是否已经计算了邻接矩阵
            if 'neighbors' not in adata.uns:
                sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
            
            sc.tl.umap(adata)
            sc.tl.leiden(adata, resolution=resolution)
            sc.tl.paga(adata)
            if adata.shape[0] < max_cell_for_subsampling:
                sub_adata_x = adata.X
            else:
                indices = np.arange(adata.shape[0])
                selected_ind = np.random.choice(indices, max_cell_for_subsampling, False)
                sub_adata_x = adata.X[selected_ind, :]
            sum_dists = distance_matrix(sub_adata_x, sub_adata_x).sum(axis=1)
            adata.uns['iroot'] = np.argmax(sum_dists)
            sc.tl.diffmap(adata)
            sc.tl.dpt(adata)
            pSM_values = adata.obs['dpt_pseudotime'].to_numpy()
            save_dir = os.path.dirname(pSM_values_save_filepath)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.savetxt(pSM_values_save_filepath, pSM_values, fmt='%.5f', header='', footer='', comments='')
            print(f"pseudo-Spatiotemporal Map(pSM) calculation complete, pSM values of cells or spots saved at {pSM_values_save_filepath}!")
            self.pSM_values = pSM_values
        except NameError:
            print(error_message)
        except AttributeError:
            print(error_message)
    def plot_pSM(self, pSM_figure_save_filepath="./pseudo-Spatiotemporal-Map.pdf", colormap='roma', scatter_sz=1., rsz=4., csz=4., wspace=.4, hspace=.5, left=0.125, right=0.9, bottom=0.1, top=0.9):
        """
        Plot the domain segmentation for ST data in spatial
        :param pSM_figure_save_filepath: the default save path for the figure
        :type pSM_figure_save_filepath: class:`str`, optional, default: "./Spatiotemporal-Map.pdf"
        :param colormap: The colormap to use. See `https://www.fabiocrameri.ch/colourmaps-userguide/` for name list of colormaps
        :type colormap: str, optional, default: roma
        :param scatter_sz: The marker size in points**2
        :type scatter_sz: float, optional, default: 1.0
        :param rsz: row size of the figure in inches, default: 4.0
        :type rsz: float, optional
        :param csz: column size of the figure in inches, default: 4.0
        :type csz: float, optional
        :param wspace: the amount of width reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
        :type wspace: float, optional
        :param hspace: the amount of height reserved for space between subplots, expressed as a fraction of the average axis width, default: 0.4
        :type hspace: float, optional
        :param left: the leftmost position of the subplots of the figure in fraction, default: 0.125
        :type left: float, optional
        :param right: the rightmost position of the subplots of the figure in fraction, default: 0.9
        :type right: float, optional
        :param bottom: the bottom position of the subplots of the figure in fraction, default: 0.1
        :type bottom: float, optional
        :param top: the top position of the subplots of the figure in fraction, default: 0.9
        :type top: float, optional
        """
        error_message = "No pseudo Spatiotemporal Map data found, please ensure you have run the pseudo_Spatiotemporal_Map() method."
        try:
            fig, ax = self.prepare_figure(rsz=rsz, csz=csz, wspace=wspace, hspace=hspace, left=left, right=right, bottom=bottom, top=top)
            x, y = self.adata_preprocessed.obsm["spatial"][:, 0], self.adata_preprocessed.obsm["spatial"][:, 1]
            st = ax.scatter(x, y, s=scatter_sz, c=self.pSM_values, cmap=f"cmc.{colormap}", marker=".")
            ax.invert_yaxis()
            clb = fig.colorbar(st)
            clb.ax.set_ylabel("pseudotime", labelpad=10, rotation=270, fontsize=10, weight='bold')
            ax.set_title("pseudo-Spatiotemporal Map", fontsize=14)
            ax.set_facecolor("none")

            save_dir = os.path.dirname(pSM_figure_save_filepath)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(pSM_figure_save_filepath, dpi=300)
            print(f"Plotting complete, pseudo-Spatiotemporal Map figure saved at {pSM_figure_save_filepath} !")
            plt.close('all')
        except NameError:
            print(error_message)
        except AttributeError:
            print(error_message)