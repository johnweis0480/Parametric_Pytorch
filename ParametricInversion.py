
import SimPEG.potential_fields as PF
from SimPEG import (
    utils, maps,data_misfit,directives,regularization,optimization,inversion,inverse_problem
)
import matplotlib.pyplot as plt
import numpy as np
from discretize import TreeMesh
import discretize
import warnings
warnings.filterwarnings("ignore")
import torch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

#Utility Function to plot ellipsoid
def plot_ellipse(u,v,a,b,angle):
    t = np.linspace(0, 2 * np.pi, 100)
    x_points = a * np.cos(t)
    y_points = b * np.sin(t)
    angle_plot = np.radians(-angle)
    R = np.array([[np.cos(angle_plot), -np.sin(angle_plot)], [np.sin(angle_plot), np.cos(angle_plot)]])
    test = np.vstack((x_points, y_points))
    points = R @ test
    x = u + points[0,:]
    y = v + points[1,:]
    return x,y


#Parametric Ellipse function, could be any parameterization, needs to be written in pytorch
def torch_transform(m, params, xyz):
    xyz = xyz
    c = params[0]
    n_ellipse = params[1]
    p_0 = 0

    X = torch.tensor(xyz[:, 0])
    Y = torch.tensor(xyz[:, 1])
    Z = torch.tensor(xyz[:, 2])

    xyz = torch.vstack((X, Y, Z))


    for i in range(n_ellipse):
        rx, ry, rz, phix, phiy, phiz, x_0, y_0, z_0, p_1, a = m[i * 11:(i + 1) * 11]

        xyz_0 = torch.vstack((x_0, y_0, z_0))

        S = torch.zeros((3, 3), dtype=torch.float64)
        S[0, 0] = 2 / rx
        S[1, 1] = 2 / ry
        S[2, 2] = 2 / rz

        Rx = torch.zeros_like(S)
        Rx[0, 0] = 1
        Rx[1, 1] = torch.cos(phix)
        Rx[1, 2] = -torch.sin(phix)
        Rx[2, 2] = torch.cos(phix)
        Rx[2, 1] = torch.sin(phix)

        Ry = torch.zeros_like(S)
        Ry[1, 1] = 1
        Ry[0, 0] = torch.cos(phiy)
        Ry[2, 0] = -torch.sin(phiy)
        Ry[2, 2] = torch.cos(phiy)
        Ry[0, 2] = torch.sin(phiy)

        Rz = torch.zeros_like(S)
        Rz[2, 2] = 1
        Rz[0, 0] = torch.cos(phiz)
        Rz[0, 1] = -torch.sin(phiz)
        Rz[1, 1] = torch.cos(phiz)
        Rz[1, 0] = torch.sin(phiz)

        T = S @ Rx @ Ry @ Rz
        M = T.T @ T

        xyz_m_xyz_0 = xyz - xyz_0
        tau = M @ (xyz_m_xyz_0)

        tau = c - torch.sum(xyz_m_xyz_0 * tau, dim=0)

        p = p_0 + .5 * (1 + torch.tanh(a * tau)) * (p_1 - p_0)

        if i ==0:
            num = p * torch.exp(p)
            den = torch.exp(p)
        else:
            num = num+(p * torch.exp(p))
            den = den+ (torch.exp(p))
    p = num / den
    return p

#Load Data and mesh
mesh = TreeMesh.read_UBC('Mesh.ubc')
dpred_mag = utils.io_utils.read_mag3d_ubc('magnetic_data.obs')
survey_mag = dpred_mag.survey



#Set active indices
ind_active_block = utils.model_builder.getIndicesBlock(np.array([-3000,-3000,-3000]),np.array([3000,3000,0]),mesh.cell_centers)
ind_active = np.zeros(mesh.n_cells,dtype='bool')
ind_active[ind_active_block] = 1
nC = int(ind_active.sum())



#Create ellipsoidal model
#radius x,y,z angle x,y,z initial location x,y,z, susceptility
m = np.array([200.0,200.0,200,0.0,0.0,0.0,-250.0,00.0,-400.0,2.0,1.0])
#m = np.load('010-singlebody_plate_noisefloor-2023-07-14-15-13.npy')
#Other fixed parameters (another smoothness parameter, and the number of ellipsoids)
params = [5,1]

#Derivative gets computed automatically given a forward transform function defined above
torch_map = maps.PytorchMapping(mesh=mesh,nP=11,indicesActive=ind_active,forward_transform=torch_transform,params=params)

#Other mappings
chiMap = maps.ChiMap(mesh)
active_map = maps.InjectActiveCells(mesh,ind_active,0)



#Mappings automatically take derivatives associated with mappings, * means they are composed in order in simpeg
total_map = chiMap*active_map*torch_map

#Plot initial model
fig = plt.figure(figsize=(9, 4))
ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
yplot = 0
plot_ind = int(np.argmin(abs(yplot-mesh.cell_centers_y)))

model_sus = total_map*m

model_plot = active_map*torch_map*m

colors = ['deeppink','red','orange','yellow','springgreen','aqua','white']
cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
cmap1_r = cmap1.reversed()
mpl.colormaps.register(cmap=cmap1)
mpl.colormaps.register(cmap=cmap1_r)

mesh.plotSlice(
    model_plot,
    normal="Y",
    ax = ax1,
    range_x = (-1800,1800),
    range_y = (-1400,00),
    ind=plot_ind,
    grid=False,
    pcolor_opts={"cmap": "mycmap_r"},
    clim=(np.min(model_plot), np.max(model_plot)),
)

plt.gca().set_aspect('equal', adjustable='box')
ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
norm = mpl.colors.Normalize(vmin=np.min(model_plot), vmax=np.max(model_plot))
cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical",cmap = cmap1_r)
cbar.set_label("Magnetic Susceptibility (SI)", rotation=270, labelpad=15, size=12)

#Plot Ellipse Outline
center = np.array([0,0,-500.0])
axes = [500.0,100.0]
strike_dip_rake = [0,45,90]
xdm,ydm = plot_ellipse(center[0],center[2],axes[0],axes[1],strike_dip_rake[1])
ax1.plot(xdm,ydm,linewidth=3,c='blue')
plt.show()


#Set simulation
simulation_mag = PF.magnetics.simulation.Simulation3DDifferential(
    survey=survey_mag,
    mesh=mesh,
    exact_TMI=True,
    muMap=total_map,
    storeJ = True
)



rx_loc = survey_mag.receiver_locations



dmis = data_misfit.L2DataMisfit(data = dpred_mag,simulation=simulation_mag)

start_model = m

#Need to pass a dummy regularization, but can set beta to 0
reg_parametric = regularization.Tikhonov(discretize.TensorMesh([np.ones(11)]))

#Set bounds for optimization
lower_bounds = np.r_[100,100,100,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,1.0,.8,]

upper_bounds = np.r_[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,0,np.inf,2.0,]

opt = optimization.ProjectedGNCG(
    maxIter=10, lower=lower_bounds, upper=upper_bounds, maxIterLS=20, maxIterCG=50, tolCG=.001
)

inv_prob = inverse_problem.BaseInvProblem(
    dmis, reg_parametric, opt,beta=0
)

update_jacobi = directives.UpdatePreconditioner(update_every_iteration = True)

directives_list = [
    update_jacobi,
]


inv = inversion.BaseInversion(
    inv_prob, directiveList=directives_list
)

recovered_model = inv.run(start_model)

fig = plt.figure(figsize=(9, 4))
ax1 = fig.add_axes([0.1, 0.12, 0.73, 0.78])
yplot = 0

model_plot = active_map*torch_map*recovered_model

mesh.plotSlice(
    model_plot,
    normal="Y",
    ax = ax1,
    range_x = (-1800,1800),
    range_y = (-1400,00),
    ind=plot_ind,
    grid=False,
    pcolor_opts={"cmap": "mycmap_r"},
    clim=(np.min(model_plot), np.max(model_plot)),
)

plt.gca().set_aspect('equal', adjustable='box')
ax2 = fig.add_axes([0.85, 0.12, 0.05, 0.78])
norm = mpl.colors.Normalize(vmin=np.min(model_plot), vmax=np.max(model_plot))
cbar = mpl.colorbar.ColorbarBase(ax2, norm=norm, orientation="vertical",cmap = cmap1_r)
cbar.set_label("Magnetic Susceptibility (SI)", rotation=270, labelpad=15, size=12)

ax1.plot(xdm,ydm)

plt.show()



