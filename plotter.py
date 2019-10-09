import matplotlib.pyplot as plt

var1 = 'lep1_pt'
var2 = 'lep2_pt'
var3 = 'reco_zv_mass'

def plotVars(x_train, pandaDF, variables, label_names):
    fig = plt.figure(figsize=(12, 9))
    # label
    fig.suptitle('VV training dataset')
    subPlots={}
    for var_idx in range(len(variables)):
        subPlots[var_idx] = fig.add_subplot(2, 3, var_idx+1)
        subPlots[var_idx].ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        
        subPlots[var_idx].set_ylabel('Normalised Event')
        subPlots[var_idx].set_xlabel(variables[var_idx])
        for t in range(2):
            x=[]
            for i in range(len(x_train)):
                if pandaDF['isSignal'][i] == t:
                    x.append(x_train[i,var_idx])
            n, bins, patches = subPlots[var_idx].hist(x, 20, density=1, alpha=0.5, label=label_names[t])
        subPlots[var_idx].legend()
    plt.savefig('dist_1d.pdf')
    pass

def plotTruth(x_train, pandaDF, label_names):
    fig = plt.figure(figsize=(12, 9))
    
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    subPlots=[ax1,ax2,ax3]
    for t in range(2):
        x=[]
        y=[]
        z=[]
        for i in range(len(x_train)):
            if pandaDF['isSignal'][i] == t:
                x.append(x_train[i,0])
                y.append(x_train[i,1])
                z.append(x_train[i,2])
        cm = [plt.cm.Paired([c]) for c in [0,6]]
        ax1.scatter(x, y, c=cm[t], edgecolors='none',alpha=0.5, label=label_names[t])
        ax2.scatter(x, z, c=cm[t], edgecolors='none',alpha=0.5, label=label_names[t])
        ax3.scatter(y, z, c=cm[t], edgecolors='none',alpha=0.5, label=label_names[t])
    
    for pl in subPlots:
        pl.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        pl.legend()
    ax1.set_xlabel(var1)
    ax1.set_ylabel(var2)
    ax2.set_xlabel(var1)
    ax2.set_ylabel(var3)
    ax3.set_xlabel(var2)
    ax3.set_ylabel(var3)
 #   plt.legend()
    #plt.show()
    plt.savefig('Truth_3d_2x2.pdf')

# Plot with initial parameter
#decision_boundary(x_train, y_train_label, theta_init, title='Initial')

# Plot with optimized parameter
#decision_boundary(X, y_test_label, theta_opt, title='Optimized')
