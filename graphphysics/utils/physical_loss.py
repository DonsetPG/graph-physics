import torch
from torch_geometric.data import Data

from graphphysics.utils.vectorial_operators import compute_divergence, compute_gradient


class PhysicalLoss:
    """
    Linear momentum residual:
        rho * a - div(sigma)
    with:
        sigma = p I - 2 mu dev(eps),
        eps = 0.5 * (grad u + grad u^T),
        dev(eps) = eps - tr(eps)/3 * I
    """

    def __init__(self, gradient_method: str = "least_squares"):
        self.gradient_method = gradient_method

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compute_deviatoric_part(self, tensor: torch.Tensor) -> torch.Tensor:
        """Deviatoric part """
        I = torch.eye(3, device=tensor.device, dtype=tensor.dtype).unsqueeze(0)
        trace = tensor.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
        return tensor - (trace / 3.0).unsqueeze(-1) * I

    def compute_stress(
        self, pressure: torch.Tensor, mu: torch.Tensor, dev_strain: torch.Tensor
    ) -> torch.Tensor:
        """sigma = p I + 2 mu dev(eps)."""
        p = pressure.view(-1, 1, 1).to(dev_strain.device, dev_strain.dtype)
        mu = mu.view(-1, 1, 1).to(dev_strain.device, dev_strain.dtype)
        I = torch.eye(3, device=dev_strain.device, dtype=dev_strain.dtype).unsqueeze(0)
        return p * I + 2.0 * mu * dev_strain

    def compute_strain_from_displacement(
        self,
        graph: Data,
        displacement: torch.Tensor | None = None,
        method: str = "finite_diff",
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute strain from displacementtt
        """
        method = method
        device = device or graph.x.device

        field = displacement
        if field is None:
            field = graph.x[:, 0:3]
        field = field.to(device)

        grad_u = compute_gradient(graph, field, method=method, device=device)
        strain = 0.5 * (grad_u + grad_u.transpose(1, 2))
        dev_strain = self.compute_deviatoric_part(strain)
        return strain, dev_strain

    def _divergence_of_stress(
        self,
        graph: Data,
        stress: torch.Tensor,
        method: str = "finite_diff",
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        method = method or self.gradient_method
        device = device or graph.x.device
        components = []
        for i in range(3):
            components.append(
                compute_divergence(
                    graph,
                    stress[:, i, :],
                    method=method,
                    device=device,
                )
            )
        return torch.stack(components, dim=1)

    def forward(
        self,
        graph: Data,
        network_output_physical: torch.Tensor,
        current_physical: torch.Tensor,
        previous_physical: torch.Tensor,
        rho: torch.Tensor,
        mu: torch.Tensor,
        pressure: torch.Tensor,
        target_dt: float,
        is_acceleration: bool = True,
    ) -> torch.Tensor:
        # Acceleration term (second-order finite difference)
        if is_acceleration:
            accel = (
                network_output_physical - 2.0 * current_physical + previous_physical
            ) / (target_dt**2)
        else:
            accel = 0

        # Stress divergence
        _, dev_strain = self.compute_strain_from_displacement(
            graph, displacement=current_physical
        )
        stress = self.compute_stress(pressure, mu, dev_strain)
        div_stress = self._divergence_of_stress(graph, stress)

        # Residual: rho * u_ddot - div(sigma)
        rho = rho.view(-1, 1).to(network_output_physical.device, network_output_physical.dtype)
        residual = rho * accel - div_stress

        # print(f"rho {rho}")
        # print(f"accel {accel}")
        # print(f"div_stress {div_stress}")

        return torch.norm(residual, dim=1)

class ResidualLoss:
    """
    Quasi-static equilibrium residual:

        r(u) = div(sigma(u))

    eps(u)   = 0.5 * (grad u + grad u^T)
    sigma(u)= lambda * tr(eps) * I + 2 * mu * eps
    """

    def __init__(
        self,
        gradient_method: str = "least_squares",
        lam: float = 0.0,
        mu: float = 1.0,
    ):
        self.gradient_method = gradient_method
        self.lam = lam
        self.mu = mu

    def __call__(self, graph: Data, displacement: torch.Tensor) -> torch.Tensor:
        return self.forward(graph, displacement)

    def forward(
        self,
        graph: Data,
        displacement: torch.Tensor,   # [N,3]
    ) -> torch.Tensor:
        device = displacement.device

        # grad u : [N,3,3]
        grad_u = compute_gradient(
            graph,
            displacement,
            method=self.gradient_method,
            device=device,
        )

        # strain
        eps = 0.5 * (grad_u + grad_u.transpose(1, 2))

        # trace(eps)
        tr_eps = eps.diagonal(dim1=-2, dim2=-1).sum(-1)  # [N]

        # identity
        I = torch.eye(3, device=device, dtype=eps.dtype).unsqueeze(0)

        # stress
        sigma = (
            self.lam * tr_eps.view(-1, 1, 1) * I
            + 2.0 * self.mu * eps
        )  # [N,3,3]

        # div(sigma)
        div_sigma = []
        for i in range(3):
            div_sigma.append(
                compute_divergence(
                    graph,
                    sigma[:, i, :],   # row i
                    method=self.gradient_method,
                    device=device,
                )
            )

        div_sigma = torch.stack(div_sigma, dim=1)  # [N,3]

        # node-wise residual norm
        residual = torch.norm(div_sigma, dim=1)    # [N]

        return residual

class HyperelasticResidual:
    """
    Quasi-static hyperelastic residual using ONLY deformation (displacement u):

        F = I + grad(u)
        P = mu * J^(-2/3) * (F - (I1/3) * F^{-T}) + kappa * ln(J) * F^{-T}
        r = Div(P)

    This is a standard compressible Neo-Hookean deviatoric/volumetric split.
    Returns node-wise residual norm ||Div(P)||.

    Assumptions:
      - grad(u) is computed w.r.t. the reference configuration (Lagrangian).
      - compute_gradient returns [N,3,3] and compute_divergence maps [N,3] -> [N].
    """

    def __init__(
        self,
        gradient_method: str = "finite_diff",
        mu: float = 1,
        kappa: float = 2.0,
        eps: float = 1e-12,
    ):
        self.gradient_method = gradient_method
        self.mu = mu
        self.kappa = kappa
        self.eps = eps

    def __call__(self, graph: Data, displacement: torch.Tensor) -> torch.Tensor:
        return self.forward(graph, displacement)

    def forward(self, graph: Data, displacement: torch.Tensor) -> torch.Tensor:
        device = displacement.device
        dtype = displacement.dtype

        # grad u : [N,3,3]
        grad_u = compute_gradient(
            graph, displacement, method=self.gradient_method, device=device
        )

        # F = I + grad u : [N,3,3]
        I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
        F = I + grad_u

        # J = det(F) : [N]
        J = torch.linalg.det(F).clamp_min(self.eps)

        # F^{-T} : [N,3,3]
        FinvT = torch.linalg.inv(F).transpose(1, 2)

        # C = F^T F, I1 = tr(C)
        C = F.transpose(1, 2) @ F
        I1 = C.diagonal(dim1=-2, dim2=-1).sum(-1)  # [N]

        # Deviatoric scaling
        Jm23 = J.pow(-2.0 / 3.0)  # [N]

        mu = torch.as_tensor(self.mu, device=device, dtype=dtype)
        kappa = torch.as_tensor(self.kappa, device=device, dtype=dtype)

        # P = mu*J^{-2/3}*(F - (I1/3)F^{-T}) + kappa*ln(J)*F^{-T}
        P_dev = mu * Jm23.view(-1, 1, 1) * (
            F - (I1 / 3.0).view(-1, 1, 1) * FinvT
        )
        P_vol = kappa * torch.log(J).view(-1, 1, 1) * FinvT
        P = P_dev + P_vol  # [N,3,3]

        # Div(P) : [N,3]
        divP = []
        for i in range(3):
            divP.append(
                compute_divergence(
                    graph,
                    P[:, i, :],  # row i
                    method=self.gradient_method,
                    device=device,
                )
            )
        divP = torch.stack(divP, dim=1)  # [N,3]

        # node-wise residual norm
        return torch.norm(divP, dim=1)  # [N]
