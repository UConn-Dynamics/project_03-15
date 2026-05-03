### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDynamicsCore")].get_my_display_value(@__MODULE__, $(Expr(:quote, def))); catch; missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv
        el
    end
    #! format: on
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
begin
	import Pkg
	Pkg.activate(temp = true)
	Pkg.add(["Plots", "PlutoUI", "LinearAlgebra", "Printf"])
end

# ╔═╡ a1b2c3d4-0001-11f1-0000-000000000001
using LinearAlgebra, Plots, PlutoUI, Printf

# ╔═╡ f17103ea-06bf-11f1-a2b0-79e68ed152eb
md"""
# Project\_03 - Multibody Dynamic modeling

![Sliding compound pendulum with a support block connected to a spring and rotating compound pendulum](https://raw.githubusercontent.com/cooperrc/me5180-project_02/refs/heads/main/spring_compound-2_bodies.png)

In this project, a rigid bar is connected to a sliding block along a
horizontal tracks. The sliding block is connected to a spring that stretches and compresses. The rigid bar ``L = 0.4~\text{m}`` acts as a compound pendulum.

1. ``x_1``-``y_1``- describes block 1 position and orientation, ``\theta_1``
2. ``x_2``-``y_2``- describes the rigid bar position and orientation, ``\theta_2``

The applied forces are,

1. Spring attached to block 1, ``F = -k x_1`` where ``k = 10~\text{N/m}``
2. gravity acting on block 1 and the rigid bar, ``F_1 = -m_1g\hat{j}`` and ``F_2 = -m_2 g\hat{j}`` where ``m_1 = 0.1~\text{kg}`` and ``m_2 = 0.3~\text{kg}``

In this project, you need to

1. determine constraint equations ``C(\mathbf{q},~t)``
2. Create an augmented solution method for the dynamic motion of these two moving parts
3. visualize the motion of the system as the two parts complete at least one oscillation
4. calculate and show (graph or vectors) the constraint forces acting on the 2-body system

---

### Generalised coordinates
```
q = [x₁, y₁, θ₁, x₂, y₂, θ₂]
```
- ``x_1, y_1, \theta_1`` — block position and orientation
- ``x_2, y_2, \theta_2`` — bar centre-of-mass position and orientation

### Constraints ``C(\mathbf{q}) = 0``
| # | Equation | Meaning |
|---|----------|---------|
| 1 | ``y_1 = 0`` | block on horizontal track |
| 2 | ``\theta_1 = 0`` | block does not rotate |
| 3 | ``x_2 - x_1 - \tfrac{L}{2}\sin\theta_2 = 0`` | pin joint (x) |
| 4 | ``y_2 - \tfrac{L}{2}\cos\theta_2 = 0`` | pin joint (y) |

### Augmented (Lagrange multiplier) formulation

```math
\begin{bmatrix} M & C_q^T \\ C_q & 0 \end{bmatrix}
\begin{bmatrix} \ddot{q} \\ \lambda \end{bmatrix}
=
\begin{bmatrix} Q_e \\ \gamma \end{bmatrix}
```

where ``\lambda`` gives the **constraint forces** directly.
"""

# ╔═╡ a1b2c3d4-0002-11f1-0000-000000000002
md"## ① System parameters"

# ╔═╡ a1b2c3d4-0003-11f1-0000-000000000003
md"""
Use the sliders to explore different parameter values.

**Bar length** ``L`` (m): $(@bind L_val Slider(0.1:0.05:1.0, default=0.4, show_value=true))

**Spring stiffness** ``k`` (N/m): $(@bind k_val Slider(1.0:1.0:50.0, default=10.0, show_value=true))

**Block mass** ``m_1`` (kg): $(@bind m1_val Slider(0.05:0.05:1.0, default=0.1, show_value=true))

**Bar mass** ``m_2`` (kg): $(@bind m2_val Slider(0.05:0.05:1.0, default=0.3, show_value=true))
"""

# ╔═╡ a1b2c3d4-0004-11f1-0000-000000000004
md"## ② Initial conditions"

# ╔═╡ a1b2c3d4-0005-11f1-0000-000000000005
md"""
**Initial block displacement** ``x_1(0)`` (m): $(@bind x1_ic Slider(-0.2:0.01:0.2, default=0.05, show_value=true))

**Initial bar angle** ``\theta_2(0)`` (°): $(@bind θ2_ic_deg Slider(-60:5:60, default=30, show_value=true))

**Simulation duration** (s): $(@bind t_end_val Slider(2.0:1.0:20.0, default=10.0, show_value=true))
"""

# ╔═╡ a1b2c3d4-0006-11f1-0000-000000000006
md"## ③ Solver"

# ╔═╡ a1b2c3d4-0007-11f1-0000-000000000007
begin
	# ── Derived constants from sliders ──────────────────────────────────────────
	const g_const = 9.81
	I2_val = m2_val * L_val^2 / 12   # uniform bar moment of inertia

	md"""
	**Derived:** ``I_2 = m_2 L^2 / 12 =`` $(round(I2_val, sigdigits=5)) kg·m²
	"""
end

# ╔═╡ a1b2c3d4-0008-11f1-0000-000000000008
begin
	# ── Mass matrix ─────────────────────────────────────────────────────────────
	function mass_matrix(m1, m2, I2)
		M = zeros(6, 6)
		M[1,1] = m1;  M[2,2] = m1;  M[3,3] = 1e-9
		M[4,4] = m2;  M[5,5] = m2;  M[6,6] = I2
		return M
	end

	# ── External forces ─────────────────────────────────────────────────────────
	function Qe(q, m1, m2, k)
		F = zeros(6)
		F[1] = -k * q[1]
		F[2] = -m1 * g_const
		F[5] = -m2 * g_const
		return F
	end

	# ── Constraint vector ────────────────────────────────────────────────────────
	function C_vec(q, L)
		x1,y1,θ1,x2,y2,θ2 = q
		return [y1,
		        θ1,
		        x2 - x1 - (L/2)*sin(θ2),
		        y2 - (L/2)*cos(θ2)]
	end

	# ── Constraint Jacobian ──────────────────────────────────────────────────────
	function Cq_mat(q, L)
		θ2 = q[6]
		J = zeros(4, 6)
		J[1, 2] = 1.0
		J[2, 3] = 1.0
		J[3, 1] = -1.0;  J[3, 4] = 1.0;  J[3, 6] = -(L/2)*cos(θ2)
		J[4, 5] = 1.0;   J[4, 6] =  (L/2)*sin(θ2)
		return J
	end

	# ── Gamma vector (quadratic velocity terms) ──────────────────────────────────
	function gamma_vec(q, qdot, L)
		θ2    = q[6]
		θ2dot = qdot[6]
		return [0.0, 0.0,
		        -(L/2)*sin(θ2)*θ2dot^2,
		        -(L/2)*cos(θ2)*θ2dot^2]
	end

	# ── Baumgarte-stabilised RHS ─────────────────────────────────────────────────
	const ω_stab = 10.0
	function gamma_stab(q, qdot, L)
		J  = Cq_mat(q, L)
		γ  = gamma_vec(q, qdot, L)
		return γ .- 2ω_stab .* (J * qdot) .- ω_stab^2 .* C_vec(q, L)
	end

	# ── Augmented EOM solve ──────────────────────────────────────────────────────
	function solve_aug(q, qdot, m1, m2, k, L, I2)
		M   = mass_matrix(m1, m2, I2)
		J   = Cq_mat(q, L)
		nc  = size(J, 1); nq = size(J, 2)
		A   = [M        J';
		       J  zeros(nc, nc)]
		rhs = [Qe(q, m1, m2, k); gamma_stab(q, qdot, L)]
		sol = A \ rhs
		return sol[1:nq], sol[nq+1:end]
	end

	# ── RK4 step ────────────────────────────────────────────────────────────────
	function rk4_step(q, qdot, dt, m1, m2, k, L, I2)
		function D(q_, v_)
			qdd, _ = solve_aug(q_, v_, m1, m2, k, L, I2)
			return v_, qdd
		end
		k1q,k1v = D(q, qdot)
		k2q,k2v = D(q .+ 0.5dt.*k1q, qdot .+ 0.5dt.*k1v)
		k3q,k3v = D(q .+ 0.5dt.*k2q, qdot .+ 0.5dt.*k2v)
		k4q,k4v = D(q .+ dt.*k3q,    qdot .+ dt.*k3v)
		q_new    = q    .+ (dt/6).*(k1q .+ 2k2q .+ 2k3q .+ k4q)
		qdot_new = qdot .+ (dt/6).*(k1v .+ 2k2v .+ 2k3v .+ k4v)
		return q_new, qdot_new
	end

	md"*Solver functions defined ✓*"
end

# ╔═╡ a1b2c3d4-0009-11f1-0000-000000000009
begin
	# ── Run simulation ───────────────────────────────────────────────────────────
	θ2_ic = deg2rad(θ2_ic_deg)
	q0 = [x1_ic,
	      0.0,
	      0.0,
	      x1_ic + (L_val/2)*sin(θ2_ic),
	      (L_val/2)*cos(θ2_ic),
	      θ2_ic]
	qdot0 = zeros(6)

	dt  = 1e-3
	N   = round(Int, t_end_val / dt)

	times    = Vector{Float64}(undef, N+1)
	qs_arr   = Matrix{Float64}(undef, N+1, 6)
	λs_arr   = Matrix{Float64}(undef, N+1, 4)
	C_viol   = Vector{Float64}(undef, N+1)

	q_cur    = copy(q0)
	qdot_cur = copy(qdot0)

	for i = 1:N+1
		times[i]      = (i-1)*dt
		qs_arr[i, :]  = q_cur
		_, λ          = solve_aug(q_cur, qdot_cur, m1_val, m2_val, k_val, L_val, I2_val)
		λs_arr[i, :]  = λ
		C_viol[i]     = norm(C_vec(q_cur, L_val))
		if i <= N
			q_cur, qdot_cur = rk4_step(q_cur, qdot_cur, dt,
			                            m1_val, m2_val, k_val, L_val, I2_val)
		end
	end

	x1_t  = qs_arr[:, 1]
	θ2_t  = rad2deg.(qs_arr[:, 6])
	Fx_t  = λs_arr[:, 3]
	Fy_t  = λs_arr[:, 4]

	md"**Simulation complete** — $(N) steps, max constraint violation = $(round(maximum(C_viol), sigdigits=3))"
end

# ╔═╡ a1b2c3d4-0010-11f1-0000-000000000010
md"## ④ Results – Motion"

# ╔═╡ a1b2c3d4-0011-11f1-0000-000000000011
begin
	p1 = plot(times, x1_t,
		label = "x₁ (m)",
		xlabel = "Time (s)", ylabel = "Position (m)",
		title = "Block position x₁(t)",
		linewidth = 2, color = :royalblue, legend = :topright)
	hline!(p1, [0.0], linestyle=:dash, color=:grey, label="equilibrium")

	p2 = plot(times, θ2_t,
		label = "θ₂ (°)",
		xlabel = "Time (s)", ylabel = "Angle (°)",
		title = "Bar angle θ₂(t)",
		linewidth = 2, color = :crimson, legend = :topright)
	hline!(p2, [0.0], linestyle=:dash, color=:grey, label="vertical")

	plot(p1, p2, layout=(2,1), size=(800, 500))
end

# ╔═╡ a1b2c3d4-0012-11f1-0000-000000000012
md"## ⑤ Results – Constraint (Pin-Joint) Forces"

# ╔═╡ a1b2c3d4-0013-11f1-0000-000000000013
begin
	p3 = plot(times, Fx_t,
		label = "Fₓ (N)", xlabel = "Time (s)", ylabel = "Force (N)",
		title = "Pin constraint force – x component",
		linewidth = 2, color = :darkorange)

	p4 = plot(times, Fy_t,
		label = "Fy (N)", xlabel = "Time (s)", ylabel = "Force (N)",
		title = "Pin constraint force – y component",
		linewidth = 2, color = :darkgreen)

	plot(p3, p4, layout=(2,1), size=(800, 500))
end

# ╔═╡ a1b2c3d4-0014-11f1-0000-000000000014
md"## ⑥ Phase portraits"

# ╔═╡ a1b2c3d4-0015-11f1-0000-000000000015
begin
	x1dot_t = [qs_arr[i,1] == qs_arr[min(i+1,N+1),1] ? 0.0 :
	           (qs_arr[min(i+1,N+1),1] - qs_arr[i,1])/dt for i in 1:N+1]
	θ2dot_t = [(qs_arr[min(i+1,N+1),6] - qs_arr[i,6])/dt for i in 1:N+1]

	ph1 = plot(x1_t, x1dot_t,
		xlabel = "x₁ (m)", ylabel = "ẋ₁ (m/s)",
		title = "Phase portrait – Block",
		linewidth = 1.5, color = :royalblue, legend = false, aspect_ratio = :none)

	ph2 = plot(θ2_t, rad2deg.(θ2dot_t),
		xlabel = "θ₂ (°)", ylabel = "θ̇₂ (°/s)",
		title = "Phase portrait – Bar",
		linewidth = 1.5, color = :crimson, legend = false, aspect_ratio = :none)

	plot(ph1, ph2, layout=(1,2), size=(800, 380))
end

# ╔═╡ a1b2c3d4-0016-11f1-0000-000000000016
md"## ⑦ Constraint violation (numerical health check)"

# ╔═╡ a1b2c3d4-0017-11f1-0000-000000000017
plot(times, C_viol,
	xlabel = "Time (s)", ylabel = "|C(q)| (m)",
	title = "Constraint violation (Baumgarte stabilisation, ω = $(ω_stab) rad/s)",
	linewidth = 1.5, color = :purple, yscale = :log10, legend = false,
	size = (800, 280))

# ╔═╡ a1b2c3d4-0018-11f1-0000-000000000018
md"""
## ⑧ Summary table

| Quantity | Min | Max |
|---------|-----|-----|
| ``x_1`` (m) | $(round(minimum(x1_t),digits=4)) | $(round(maximum(x1_t),digits=4)) |
| ``\theta_2`` (°) | $(round(minimum(θ2_t),digits=2)) | $(round(maximum(θ2_t),digits=2)) |
| ``F_x^{pin}`` (N) | $(round(minimum(Fx_t),digits=4)) | $(round(maximum(Fx_t),digits=4)) |
| ``F_y^{pin}`` (N) | $(round(minimum(Fy_t),digits=4)) | $(round(maximum(Fy_t),digits=4)) |
| ``\|C\|`` | — | $(round(maximum(C_viol),sigdigits=3)) |
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d3"

[compat]
Plots = "~1"
PlutoUI = "~0.7"
"""

# ╔═╡ Cell order:
# ╠═00000000-0000-0000-0000-000000000001
# ╠═a1b2c3d4-0001-11f1-0000-000000000001
# ╟─f17103ea-06bf-11f1-a2b0-79e68ed152eb
# ╟─a1b2c3d4-0002-11f1-0000-000000000002
# ╟─a1b2c3d4-0003-11f1-0000-000000000003
# ╟─a1b2c3d4-0004-11f1-0000-000000000004
# ╟─a1b2c3d4-0005-11f1-0000-000000000005
# ╟─a1b2c3d4-0006-11f1-0000-000000000006
# ╠═a1b2c3d4-0007-11f1-0000-000000000007
# ╠═a1b2c3d4-0008-11f1-0000-000000000008
# ╠═a1b2c3d4-0009-11f1-0000-000000000009
# ╟─a1b2c3d4-0010-11f1-0000-000000000010
# ╠═a1b2c3d4-0011-11f1-0000-000000000011
# ╟─a1b2c3d4-0012-11f1-0000-000000000012
# ╠═a1b2c3d4-0013-11f1-0000-000000000013
# ╟─a1b2c3d4-0014-11f1-0000-000000000014
# ╠═a1b2c3d4-0015-11f1-0000-000000000015
# ╟─a1b2c3d4-0016-11f1-0000-000000000016
# ╠═a1b2c3d4-0017-11f1-0000-000000000017
# ╟─a1b2c3d4-0018-11f1-0000-000000000018
# ╟─00000000-0000-0000-0000-000000000002
