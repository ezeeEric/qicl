import matplotlib.pyplot as plt

def plotVarsInd(x_train, pandaDF, variables, label_names):
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    for var_idx in range(len(variables)):
        fig = plt.figure(figsize=(12, 9))
        # label
        fig.suptitle('VV training dataset')
        #subPlots[var_idx] = fig.add_subplot(2, 3, var_idx+1)
        
        plt.set_ylabel('Normalised Event')
        plt.set_xlabel(variables[var_idx])
        for t in range(2):
            x=[]
            for i in range(len(x_train)):
                if pandaDF['isSignal'][i] == t:
                    x.append(x_train[i,var_idx])
            n, bins, patches = fig.hist(x, 20, density=1, alpha=0.5, label=label_names[t])
        plt.legend()
        extent = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # Pad the saved area by 10% in the x-direction and 20% in the y-direction
        fig.savefig('dist_1d_{0}.pdf'.format(variables[var_idx]), bbox_inches=extent.expanded(1.2, 1.3))
    pass

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
        extent = subPlots[var_idx].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # Pad the saved area by 10% in the x-direction and 20% in the y-direction
        fig.savefig('dist_1d_{0}.pdf'.format(variables[var_idx]), bbox_inches=extent.expanded(1.2, 1.3))
    plt.savefig('dist_1d.pdf')
    pass

def plotTruth(x_train, pandaDF, label_names):

    var1 = 'lep1_pt'
    var2 = 'lep2_pt'
    var3 = 'reco_zv_mass'
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
    count=0
    for subPlot in [ax1,ax2,ax3]:
        extent = subPlot.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # Pad the saved area by 10% in the x-direction and 20% in the y-direction
        fig.savefig('scat_{0}.pdf'.format(count), bbox_inches=extent.expanded(1.2, 1.3))
        count+=1


 #   plt.legend()
    #plt.show()
    plt.savefig('Truth_3d_2x2.pdf')

# Plot with initial parameter
#decision_boundary(x_train, y_train_label, theta_init, title='Initial')

# Plot with optimized parameter
#decision_boundary(X, y_test_label, theta_opt, title='Optimized')
