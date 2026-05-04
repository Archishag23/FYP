from pygod.generator import gen_contextual_outlier, gen_structural_outlier
from utils import gen_joint_structural_outlier


def generate_synthetic(data, synthetic_para):
    inj_anomaly_typ = synthetic_para[0]  # 'str', 'ctx', 'jnt'

    if inj_anomaly_typ == "ctx":
        n = synthetic_para[2]  # Number of nodes converting to outliers.
        k = synthetic_para[3]  # Number of candidate nodes for each outlier node.
        data, y = gen_contextual_outlier(data=data, n=n, k=k, seed=17)
        data.y = y
    elif inj_anomaly_typ == "str":
        n = synthetic_para[2]  # Number of outlier cliques.
        m = synthetic_para[3]  # Number nodes in the outlier cliques.
        data, y = gen_structural_outlier(
            data=data, n=n, m=m, p=0.2, seed=17
        )  # p: Probability of edge drop in cliques.
        data.y = y
    elif inj_anomaly_typ == "jnt":
        n = synthetic_para[2]  # Number of outlier cliques.
        m = synthetic_para[3]  # Number nodes in the outlier cliques.
        data, y = gen_joint_structural_outlier(data=data, n=n, m=m, random_state=17)
        data.y = y
    else:
        error_message = (
            f"Injecting a wrong synthetic anomaly type: {inj_anomaly_typ}. "
            f"It is supposed to be 'str', 'ctx', or 'jnt'."
        )
        raise ValueError(error_message)

    return data
