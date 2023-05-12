import dolfin as dl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def cartesian_product(array1,array2):
    dim1,n=array1.shape
    dim2,n=array2.shape
    out_arr=np.zeros((dim1*dim2,2))
    c=0
    for a in range(dim1):
        for b in range(dim2):
            out_arr[c,:]=[array1[a],array2[b]]
            c=c+1
    return out_arr
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
    potent_matrix = dl.assemble(potent_form).array()

    potential_2D = np.kron(potent_matrix, prod_matrix) + np.kron(prod_matrix, potent_matrix)
    return potential_2D

def assemble_2D_interact(interact_vect, V, N_dof, u_trial, v_test):
    interact = dl.Function(V)
    integral_dx = np.zeros((N_dof, N_dof, N_dof))
    for a in range(N_dof):
        interact.vector()[:] = np.array(interact_vect[a::N_dof])
        interact_form = interact * u_trial * v_test * dl.dx
        interact_matrix = dl.assemble(interact_form).array()
        integral_dx[:,:,a] = interact_matrix

    interact_2D = np.zeros((N_dof ** 2, N_dof ** 2))
    interact_dx = dl.Function(V)
    for i in range(N_dof):
        for j in range(N_dof):
            interact_dx.vector()[:] = integral_dx[i, j, :]
            interact_dx_form = interact_dx * u_trial * v_test * dl.dx
            interact_dx_matrix = dl.assemble(interact_dx_form).array()
            interact_2D[i * N_dof: (i + 1) * N_dof, j * N_dof: (j + 1) * N_dof] = \
                                            interact_dx_matrix
    return interact_2D

def assemble_2D_nabla_form(u_trial, v_test, prod_matrix):
    nabla_form = dl.inner(dl.grad(u_trial), dl.grad(v_test)) * dl.dx
    nabla_matrix = dl.assemble(nabla_form).array()
    nabla_2D = np.kron(nabla_matrix, prod_matrix) + np.kron(prod_matrix, nabla_matrix)
    return nabla_2D

T = 1
num_steps = 50
dt = T / num_steps
b = 1
L = 0.2
a = 10

num_elem = 20
mesh = dl.IntervalMesh(num_elem, -1, 1)

V = dl.FunctionSpace(mesh, 'P', 1)

u_trial = dl.TrialFunction(V)
v_test = dl.TestFunction(V)

dof_coords=V.tabulate_dof_coordinates()
N_dof = len(dof_coords)

global_dof = cartesian_product_coords(dof_coords,dof_coords)
N_global_dof = len(global_dof)
global_boundary_dofs = fetch_boundary_dofs(V, V, dof_coords, dof_coords)

x = global_dof[:,0]
y = global_dof[:,1]

prod_form = u_trial * v_test * dl.dx
prod_matrix = dl.assemble(prod_form).array()
prod_2D = np.kron(prod_matrix, prod_matrix)

potent_expr = dl.Expression("pow(x[0], 2) / (2 * pow(b, 2))", degree = 0, b = b)
interact_vect =  a * np.exp(-np.abs(x - y) / L)

psi0_re = np.exp(-20*((x + 0.6 ) ** 2)) + np.exp(-20*((y - 0.6) ** 2))
psi0_im = np.zeros(N_global_dof)

nabla_2D = assemble_2D_nabla_form(u_trial, v_test, prod_matrix)
potential_2D = assemble_2D_potential(potent_expr, V, u_trial, v_test, prod_matrix)
interact_2D = assemble_2D_interact(interact_vect, V, N_dof, u_trial, v_test)

hamilton_term = (nabla_2D / (2) + potential_2D + interact_2D) * dt
