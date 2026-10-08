# from torch import nn, Tensor
# from numpy import ndarray
# import torch
# import torch.nn.functional as F
# import numpy as np
# import logging
# from torch.nn.functional import softmax, cosine_similarity
# class NMFModel(nn.Module):
#     def __init__(
#         self, S: ndarray, G: ndarray, T: ndarray, d: ndarray, device: torch.device
#     ) -> None:
#         super().__init__()

#         self.S = torch.tensor(S, dtype=torch.float32, device=device)
#         self.G = torch.tensor(G, dtype=torch.float32, device=device)
#         self.T = torch.tensor(T, dtype=torch.float32, device=device)
#         self.d = torch.tensor(d, dtype=torch.float32, device=device)

#         c, g = S.shape
#         s = G.shape[0]
#         k = T.shape[1]

#         self._W = nn.Parameter(torch.rand(c, k, device=device))
#         self._V = nn.Parameter(torch.rand(s, k, device=device))
#         self._H = nn.Parameter(torch.rand(k, g, device=device))
#         self._M = nn.Parameter(torch.rand(c, s, device=device))
#     def _loss_fn(self, verbose: bool = True) -> tuple[list[Tensor], Tensor]:
#         # 非负约束
#         W = (F.softmax(self._W, dim=1) + self.T) / 2
#         V = F.softplus(self._V)
#         H = F.softplus(self._H)

#         S_pred = W @ H
#         G_pred = V @ H

#         # 计算映射矩阵M
#         M = F.softplus(self._M)

#         G_agg = M.t() @ self.S

#         loss_S = (
#             1
#             - (
#                 F.cosine_similarity(S_pred, self.S, dim=0).mean()
#                 + F.cosine_similarity(S_pred, self.S, dim=1).mean()
#             )
#             / 2
#         )

#         loss_G = (
#             1
#             - (
#                 F.cosine_similarity(G_pred, self.G, dim=0).mean()
#                 + F.cosine_similarity(G_pred, self.G, dim=1).mean()
#             )
#             / 2
#         )
#         # loss_G = (
#         #     1
#         #     - (
#         #         F.cosine_similarity(G_pred, self.G, dim=0).mean()
#         #     )
#         # )

#         # 列密度约束（通过KL散度实现）
#         current_density = M.sum(dim=0) / M.shape[0]  # 计算当前密度
#         loss_kl = F.kl_div(
#             input=torch.log(current_density + 1e-8),
#             target=self.d + 1e-8,
#             reduction="sum",
#         )

#         loss_agg = 1 - F.cosine_similarity(G_agg, self.G, dim=0).mean()

#         if verbose:
#             term_numbers = [loss.item() for loss in [loss_S, loss_G, loss_kl, loss_agg]]
#             term_names = ["loss_S", "loss_G", "loss_kl", "loss_agg"]

#             d = dict(zip(term_names, term_numbers))
#             clean_dict = {k: v for k, v in d.items() if not np.isnan(v)}
#             msg = []
#             for k, v in clean_dict.items():
#                 m = f"{k}: {v:.3f}"
#                 msg.append(m)

#             print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

#         loss_total = loss_S+loss_G+loss_kl+loss_agg
#         return [loss_total, loss_S, loss_G, loss_kl, loss_agg], M

#     def train_model(
#         self,
#         num_epochs: int,
#         learning_rate: float = 0.1,
#         print_each: int | None = 100,
#         call_back=None,
#     ) -> tuple[ndarray, dict[str, list[float]]]:
#         optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

#         if print_each:
#             logging.info(f"Printing scores every {print_each} epochs.")

#         keys = ["loss_total", "loss_S", "loss_G", "loss_kl", "loss_agg"]
#         values: list[list[float]] = [[] for _ in range(len(keys))]
#         training_history = dict(zip(keys, values))

#         for epoch in range(num_epochs + 1):
#             verbose = print_each is not None and epoch % print_each == 0
#             run_loss, M = self._loss_fn(verbose=verbose)

#             loss = run_loss[0]

#             for i in range(len(keys)):
#                 training_history[keys[i]].append(run_loss[i].item())

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             if call_back:
#                 call_back(epoch, num_epochs, training_history)
#                 # call_back(epoch, training_history)

#         with torch.no_grad():
#             output = M.cpu().numpy()
#             return output, training_history
from torch import nn, Tensor
from numpy import ndarray
import torch
import torch.nn.functional as F
import numpy as np
import logging


class NMFModel(nn.Module):
    def __init__(
        self, S: ndarray, G: ndarray, T: ndarray, d: ndarray, device: torch.device,
        reconstruction_mode: str = "shared",
        n_programs: int | None = None,
        type_prior_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if reconstruction_mode not in ("shared", "separate", "none", "fixed_random"):
            raise ValueError("Unknown reconstruction_mode")
        self.reconstruction_mode = reconstruction_mode
        if not 0 <= type_prior_weight <= 1:
            raise ValueError("type_prior_weight must be in [0, 1]")
        if n_programs is not None and (not isinstance(n_programs, (int, np.integer)) or n_programs < 1):
            raise ValueError("n_programs must be a positive integer or None (legacy)")
        self.type_prior_weight = type_prior_weight

        self.S = torch.tensor(S, dtype=torch.float32, device=device)
        self.G = torch.tensor(G, dtype=torch.float32, device=device)
        self.T = torch.tensor(T, dtype=torch.float32, device=device)
        self.d = torch.tensor(d, dtype=torch.float32, device=device)

        c, g = S.shape
        s = G.shape[0]
        # Explicit K uses a learned C-by-K type/program association. Only the
        # legacy None path identifies programs with the C cell-type columns.
        k = T.shape[1] if n_programs is None else int(n_programs)
        self.n_programs = k

        self._W = nn.Parameter(torch.rand(c, k, device=device))
        self._V = nn.Parameter(torch.rand(s, k, device=device))
        self._H = nn.Parameter(torch.rand(k, g, device=device),
                               requires_grad=reconstruction_mode not in ("none", "fixed_random"))
        if n_programs is not None and type_prior_weight > 0:
            self._A = nn.Parameter(torch.rand(T.shape[1], k, device=device))
        if reconstruction_mode == "separate":
            # Same initialization, independently trained dictionaries thereafter.
            self._H_sp = nn.Parameter(self._H.detach().clone())

    def _factor_W(self):
        if self.type_prior_weight == 0:
            return F.softmax(self._W, dim=1)
        prior = self.T @ F.softmax(self._A, dim=1) if hasattr(self, "_A") else self.T
        if self.type_prior_weight == .5:
            return (F.softmax(self._W, dim=1) + prior) / 2
        return (1 - self.type_prior_weight) * F.softmax(self._W, dim=1) + self.type_prior_weight * prior

    def export_factors(self):
        """Return final constrained factors; H is programs x training genes."""
        with torch.no_grad():
            W = self._factor_W()
            V = F.softplus(self._V)
            H = F.softplus(self._H)
            values = {"W": W, "V": V}
            if hasattr(self, "_A"):
                values["A"] = F.softmax(self._A, dim=1)
            if self.reconstruction_mode == "separate":
                values.update(H_sc=H, H_sp=F.softplus(self._H_sp))
            elif self.reconstruction_mode != "none":
                values["H"] = H
            return {name: value.cpu().numpy().copy() for name, value in values.items()}

    def _loss_fn(self, verbose: bool = True) -> tuple[list[Tensor], Tensor]:
        # 非负约束
        W = self._factor_W()
        V = F.softplus(self._V)
        H = F.softplus(self._H)

        S_pred = W @ H
        H_sp = F.softplus(self._H_sp) if self.reconstruction_mode == "separate" else H
        G_pred = V @ H_sp

        # 计算映射矩阵M
        M = F.softmax(W @ V.t(), dim=1)

        G_agg = M.t() @ self.S

        loss_S = (
            1
            - (
                F.cosine_similarity(S_pred, self.S, dim=0).mean()
                + F.cosine_similarity(S_pred, self.S, dim=1).mean()
            )
            / 2
        )

        loss_G = (
            1
            - (
                F.cosine_similarity(G_pred, self.G, dim=0).mean()
                + F.cosine_similarity(G_pred, self.G, dim=1).mean()
            )
            / 2
        )

        if self.reconstruction_mode == "none":
            # Same W/V map and density/aggregation terms; no H reconstruction signal.
            loss_S = self.S.new_zeros(())
            loss_G = self.S.new_zeros(())

        # 列密度约束（通过KL散度实现）
        current_density = M.sum(dim=0) / M.shape[0]  # 计算当前密度
        loss_kl = F.kl_div(
            input=torch.log(current_density + 1e-8),
            target=self.d + 1e-8,
            reduction="sum",
        )

        loss_agg = 1 - F.cosine_similarity(G_agg, self.G, dim=0).mean()

        if verbose:
            term_numbers = [loss.item() for loss in [loss_S, loss_G, loss_kl, loss_agg]]
            term_names = ["loss_S", "loss_G", "loss_kl", "loss_agg"]

            d = dict(zip(term_names, term_numbers))
            clean_dict = {k: v for k, v in d.items() if not np.isnan(v)}
            msg = []
            for k, v in clean_dict.items():
                m = f"{k}: {v:.3f}"
                msg.append(m)

            print(str(msg).replace("[", "").replace("]", "").replace("'", ""))

        loss_total = loss_S + loss_G + loss_kl + loss_agg
        return [loss_total, loss_S, loss_G, loss_kl, loss_agg], M

    def train_model(
        self,
        num_epochs: int,
        learning_rate: float = 0.1,
        print_each: int | None = 100,
        call_back=None,
    ) -> tuple[ndarray, dict[str, list[float]]]:
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if print_each:
            logging.info(f"Printing scores every {print_each} epochs.")

        keys = ["loss_total", "loss_S", "loss_G", "loss_kl", "loss_agg"]
        values: list[list[float]] = [[] for _ in range(len(keys))]
        training_history = dict(zip(keys, values))

        for epoch in range(num_epochs + 1):
            verbose = print_each is not None and epoch % print_each == 0
            run_loss, M = self._loss_fn(verbose=verbose)

            loss = run_loss[0]

            for i in range(len(keys)):
                training_history[keys[i]].append(run_loss[i].item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if call_back:
                call_back(epoch, num_epochs, training_history)
                # call_back(epoch, training_history)

        with torch.no_grad():
            # Recompute after the final optimizer step, matching exported factors.
            _, M = self._loss_fn(verbose=False)
            output = M.cpu().numpy()
            return output, training_history
