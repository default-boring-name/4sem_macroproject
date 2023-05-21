import dolfin as dl
import pickle as pk
import numpy as np
import time
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def cartesian_product_coords(array1,array2):
    dim1,n1=array1.shape
    dim2,n2=array2.shape
    out_arr=np.zeros((dim1*dim2,n1+n2))
    c=0
    for a in range(dim1):
        for b in range(dim2):
            out_arr[c,:]=np.append(array1[a,:],array2[b,:])
            c=c+1
    return out_arr
def cartesian_product_dofs(array1,array2):
    dim1=len(array1)
    dim2=len(array2)
    out_arr=np.zeros((dim1*dim2,2))
    c=0
    for a in range(dim1):
        for b in range(dim2):
            out_arr[c,:]=[array1[a],array2[b]]
            c=c+1
    return out_arr


def cartesian_form_to_kroneck_form(indeces, len_dim_2):
    num_indeces=indeces.shape[0]
    out_arr=np.zeros(num_indeces)
    for n in range(num_indeces):
        out_arr[n] = indeces[n,0]*len_dim_2 + indeces[n,1]
    return out_arr
def fetch_boundary_dofs(V1,V2,dof_coordinates1,dof_coordinates2):
    def boundary(x, on_boundary):
        return on_boundary

    u_D1 = dl.Expression('1.0', degree=1)

    dum1=dl.Function(V1)
    dum2=dl.Function(V2)
    bc1 = dl.DirichletBC(V1, u_D1, boundary)
    bc2 = dl.DirichletBC(V2, u_D1, boundary)

    bc1.apply(dum1.vector())
    bc2.apply(dum2.vector())

    boundary_dofs1 = np.where(dum1.vector()==1.0)[0]
    boundary_dofs2 = np.where(dum2.vector()==1.0)[0]

    boundary_coord1 = dof_coordinates1[boundary_dofs1]
    boundary_coord2 = dof_coordinates2[boundary_dofs2]

    global_boundary_dofs=np.empty((len(boundary_dofs1)*len(dof_coordinates2) + len(dof_coordinates1)*len(boundary_dofs2),2))

    ctr=0
    for j in boundary_dofs1:
        global_boundary_dofs[ctr*len(dof_coordinates2):(ctr+1)*len(dof_coordinates2),:] = \
        cartesian_product_dofs(np.array([j]),np.arange(dof_coordinates2.shape[0]))
        ctr=ctr+1

    last_ind = (ctr)*len(dof_coordinates2)


    for j in boundary_dofs2:
        global_boundary_dofs[last_ind:last_ind+len(dof_coordinates1),:] = \
        cartesian_product_dofs(np.arange(dof_coordinates1.shape[0]),np.array([j]))
        last_ind = last_ind+len(dof_coordinates1)

    global_boundary_dofs=np.unique(global_boundary_dofs,axis=0)

    global_boundary_dofs=cartesian_form_to_kroneck_form(global_boundary_dofs, len(dof_coordinates2))
    global_boundary_dofs=global_boundary_dofs.astype(int)
    return global_boundary_dofs

def assemble_2D_potential(potent_expr, V, u_trial, v_test, prod_matrix):
    potent_func = dl.interpolate(potent_expr, V)
    potent_form = potent_func * u_trial * v_test * dl.dx
    potent_matrix = sparse.dok_matrix(dl.assemble(potent_form).array())
    potential_2D = sparse.kron(potent_matrix, prod_matrix) + sparse.kron(prod_matrix, potent_matrix)
    return potential_2D

def assemble_2D_interact(interact_vect, V, N_dof, u_trial, v_test):
    interact = dl.Function(V)
    integral_dx = []
    for a in range(N_dof):
        interact.vector()[:] = np.array(interact_vect[a::N_dof])
        interact_form = interact * u_trial * v_test * dl.dx
        interact_matrix = sparse.dok_matrix(dl.assemble(interact_form).array())
        integral_dx.append(interact_matrix)

    interact_2D = np.zeros((N_dof ** 2, N_dof ** 2))
    interact_dx = dl.Function(V)
    for i in range(N_dof):
        for j in range(N_dof):

            for k in range(N_dof):
                interact_dx.vector()[k] = integral_dx[k].get((i, j), default=0)

            interact_dx_form = interact_dx * u_trial * v_test * dl.dx
            interact_dx_matrix = dl.assemble(interact_dx_form).array()
            interact_2D[i * N_dof: (i + 1) * N_dof, j * N_dof: (j + 1) * N_dof] = \
                                            interact_dx_matrix
    return sparse.dok_matrix(interact_2D)

def assemble_2D_nabla_form(u_trial, v_test, prod_matrix):
    nabla_form = dl.inner(dl.grad(u_trial), dl.grad(v_test)) * dl.dx
    nabla_matrix = sparse.dok_matrix(dl.assemble(nabla_form).array())
    nabla_2D = sparse.kron(nabla_matrix, prod_matrix) + sparse.kron(prod_matrix, nabla_matrix)
    return nabla_2D

def assemble_bilin_term(hamilton_term, prod_2D):
    bilin = sparse.bmat([[hamilton_term, prod_2D],[hamilton_term, -prod_2D]])
    return bilin

def apply_bc_hamilton_term(hamilton_term, global_boundary_dofs, N):
    Count = 0.0
    for i in global_boundary_dofs:
        non_zero = hamilton_term.getrow(i).nonzero()
        hamilton_term[i, non_zero[1]] = 0
        print(Count/ len(global_boundary_dofs))
        Count += 1;
    for i in global_boundary_dofs:
        hamilton_term[i,i] = 1

def apply_bc_bilin_term(bilinear_term, global_boundary_dofs, N):
    bilinear_term[global_boundary_dofs, :N] = 0
    bilinear_term[N + global_boundary_dofs, N:2*N] = 0
    for i in global_boundary_dofs:
        bilinear_term[i,i] = 1
        bilinear_term[N + i, N + i] = 1

def get_linear_term(psi0_re, psi0_im, prod_matrix, global_boundary_dofs, N):
    re_term = prod_matrix.dot(psi0_im)
    im_term = prod_matrix.dot(psi0_re)

    re_term[global_boundary_dofs] = 0
    im_term[global_boundary_dofs] = 0

    return np.concatenate((re_term, - im_term))

def normalize(psi_re, psi_im, N_dof):
    norm_vect = psi_re * psi_re + psi_im * psi_im
    norm_func = dl.Function(V)
    norm_dx = np.zeros((N_dof))
    for a in range(N_dof):
        norm_func.vector()[:] = np.array(norm_vect[a::N_dof])
        norm_form = norm_func * dl.dx
        norm_dx[a] = dl.assemble(norm_form)

    norm_dx_func = dl.Function(V)
    norm_dx_func.vector()[:] = norm_dx
    norm_dx_form = norm_dx_func * dl.dx
    norm = np.sqrt(dl.assemble(norm_dx_form))
    psi_re = psi_re / norm
    psi_im = psi_im / norm
    return psi_re, psi_im

def get_rho(psi_re, psi_im, N_dof):
    prob_vect = psi_re * psi_re + psi_im * psi_im
    prob_func = dl.Function(V)
    rho_1 = np.zeros((N_dof))
    rho_2 = np.zeros((N_dof))
    for a in range(N_dof):
        prob_func.vector()[:] = np.array(prob_vect[a::N_dof])
        prob_form = prob_func * dl.dx
        rho_1[a] = dl.assemble(prob_form)
    for a in range(N_dof):
        prob_func.vector()[:] = prob_vect[a * N_dof:(a + 1) * N_dof]
        prob_form = prob_func * dl.dx
        rho_2[a] = dl.assemble(prob_form)


    return rho_1 + rho_2

T = 0.1
num_steps = 50
dt = T / num_steps
m = 1
b = 1
L = 0.5
a = 30

num_elem = 10
mesh = dl.RectangleMesh(dl.Point(-1, -1), dl.Point(1, 1), num_elem, num_elem)

V = dl.FunctionSpace(mesh, 'P', 1)

u_trial = dl.TrialFunction(V)
v_test = dl.TestFunction(V)

dof_coords=V.tabulate_dof_coordinates()
N_dof = len(dof_coords)

global_dof = cartesian_product_coords(dof_coords,dof_coords)
N_global_dof = len(global_dof)
global_boundary_dofs = fetch_boundary_dofs(V, V, dof_coords, dof_coords)


x1 = global_dof[:,0]
y1 = global_dof[:,1]
x2 = global_dof[:,2]
y2 = global_dof[:,3]

prod_form = u_trial * v_test * dl.dx
prod_matrix = sparse.dok_matrix(dl.assemble(prod_form).array())
prod_2D = sparse.kron(prod_matrix, prod_matrix).todok()

potent_expr = dl.Expression("(pow(x[0], 2) + pow(x[1], 2)) / (2 * pow(b, 2))",
                            degree = 1, b=b)
interact_vect = a * np.exp(-np.abs(np.sqrt((x1 - x2)**2 + (y1 - y2)**2)) / L)
psi0_re = np.exp(-10*((x1 + 0.7 ) ** 2 + (y1 + 0.7 ) ** 2)) +\
          np.exp(-10*((x2 - 0.6 ) ** 2 + (y2) ** 2))
psi0_im = np.zeros(N_global_dof)
psi0_re, psi0_im = normalize(psi0_re, psi0_im, N_dof)

hamilton_term = assemble_2D_nabla_form(u_trial, v_test, prod_matrix) / (2 * m) * dt
hamilton_term += assemble_2D_potential(potent_expr, V, u_trial, v_test, prod_matrix)* dt
hamilton_term += assemble_2D_interact(interact_vect, V, N_dof, u_trial, v_test) * dt
hamilton_term = hamilton_term.todok()


apply_bc_hamilton_term(hamilton_term, global_boundary_dofs, N_global_dof)
hamilton_term = (hamilton_term).tocsc()
hamilton_term.eliminate_zeros()
hamilton_term = hamilton_term.todok()
bilinear_term = assemble_bilin_term(hamilton_term, prod_2D).todok()
bilinear_term = bilinear_term.tocsc()

del hamilton_term

A_solve = sparse_lin.factorized(bilinear_term)
del bilinear_term

psi = np.zeros(2 * N_global_dof)
psi_re = np.zeros(N_global_dof)
psi_im = np.zeros(N_global_dof)


rho_data = []
fig,  ax = plt.subplots()
for i in range(num_steps):
    linear_term = get_linear_term(psi0_re, psi0_im, prod_2D, global_boundary_dofs, N_global_dof)
    psi = A_solve(linear_term)
    psi_re = psi[:N_global_dof]
    psi_im = psi[N_global_dof:2*N_global_dof]
    psi_re, psi_im = normalize(psi_re, psi_im, N_dof)

    psi0_re = psi_re
    psi0_im = psi_im

    rho = get_rho(psi_re, psi_im, N_dof)
    rho_data.append(rho)
def animate(i):
       ax.clear()
       ax.tricontourf(dof_coords[:,0], dof_coords[:,1], rho_data[i], levels=100, cmap="RdBu_r")

del A_solve
ani = animation.FuncAnimation(fig, animate, num_steps, interval=150 ,blit=False)
ani.save(filename="rho_2d_new.mp4", writer="ffmpeg")
