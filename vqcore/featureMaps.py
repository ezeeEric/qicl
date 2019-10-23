from qiskit.aqua.components.feature_maps import SecondOrderExpansion,FirstOrderExpansion,PauliExpansion,PauliZExpansion,FeatureMap


def getFeatureMap(name="SecondOrderExpansion",params={}):
    fm=None
    if "SecondOrderExpansion" in name:
        fm=SecondOrderExpansion(
            feature_dimension=params["feature_dimension"], 
            depth=params["depth"],
            entanglement = params["entanglement"]
        )
    elif "FirstOrderExpansion" in name:
        fm = FirstOrderExpansion(
            feature_dimension=params["feature_dimension"], 
            depth=params["depth"],
           )
    return fm

    #PauliExpansion(
    #        feature_dimension=int(args.featMapDepth), 
    #        depth=int(args.varFormDepth),
    #)
    #PauliZExpansion(
    #        feature_dimension=int(args.featMapDepth), 
    #        depth=int(args.varFormDepth),
    #)
