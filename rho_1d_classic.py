from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def boundary(x, on_boundary):
    return on_boundary

def mesh_load():
    N = 20;
    return RectangleMesh(Point(-1,-1), Point(1,1), N, N)


def function_space_generation(mesh):
    P1 = FiniteElement("CG", "triangle", 1)
    element =  MixedElement([P1, P1])
    G = FunctionSpace(mesh, P1)
    V = FunctionSpace(mesh, element)
    return G, V

def boundary_conditions(V):
    b_func_real = Constant(0.0);
    b_func_img = Constant(0.0);
    bc_real = DirichletBC(V.sub(0), b_func_real, boundary)
    bc_img = DirichletBC(V.sub(1), b_func_img, boundary)
    return [bc_real, bc_img]

def normalize_inplace(u):
    L = (u[0] * u[0] + u[1] * u[1]) * dx
    norm = sqrt(assemble(L))
    u.assign(u / norm)

def initial_distribution_choice(V):
    u_0 = Expression(("exp(-20 * (pow(x[0] + 0.6, 2))) + exp(-20 * (pow(x[1] - 0.6, 2)))","0"),degree=2,)
    u_n = interpolate(u_0, V)
    normalize_inplace(u_n)
    return u_n

def get_rho(psi_re, psi_im, N_dof):
    prob_vect = psi_re * psi_re + psi_im * psi_im
    prob_func = Function(V)
    rho_1 = np.zeros((N_dof))
    rho_2 = np.zeros((N_dof))
    for a in range(N_dof):
        prob_func.vector()[:] = np.array(prob_vect[a::N_dof])
        prob_form = prob_func * dl.dx
        rho_1[a] = assemble(prob_form)
    for a in range(N_dof):
        prob_func.vector()[:] = prob_vect[a * N_dof:(a + 1) * N_dof]
        prob_form = prob_func * dx
        rho_2[a] = assemble(prob_form)

    return rho_1 + rho_2

T = 1
num_steps = 50
dt = T / num_steps
b = 1
a = 10
L = 0.2

G, V = function_space_generation(mesh_load())
G_dof = G.tabulate_dof_coordinates()
N_dof = len(G_dof)

v1, v2 = TestFunctions(V)
u = TrialFunction(V)
u_n = initial_distribution_choice(V)

potent_exp = Expression("(pow(x[0], 2) + pow(x[1], 2)) / (2 * pow(b, 2)) + a * exp(-abs(x[0] - x[1]) / L)", degree=1, b = b, a=a, L=L)
potential = Function(G)
potential.interpolate(potent_exp)

A = (1/(2)*(inner(grad(u[0]), grad(v1)) + inner(grad(u[1]), grad(v2))) * dt
               + potential * (u[0] * v1 + u[1] * v2) * dt +
               ((u[1])* v1 - (u[0])* v2))*dx
L =(u_n[1]* v1  - u_n[0]* v2)*dx
bcs = boundary_conditions(V)

fig,  ax = plt.subplots()
prob_data = []
U = Function(V)
for i in range(num_steps):
    solve(A == L, U, bcs)

    normalize_inplace(U)
    u_n.assign(U)

    u_re = np.zeros((N_dof))
    u_im = np.zeros((N_dof))
    u_re[:] = U.vector()[::2]
    u_im[:] = U.vector()[1::2]
    prob_data.append(u_re * u_re + u_im * u_im)
def animate_cnt(i):
       ax.clear()
       ax.tricontourf(G_dof[:,0], G_dof[:,1], prob_data[i], levels=100, cmap="RdBu_r")

ani_cnt = animation.FuncAnimation(fig, animate_cnt, num_steps, interval=50 ,blit=False)
ani_cnt.save(filename="psi_1d_classic.mp4", writer="ffmpeg")

