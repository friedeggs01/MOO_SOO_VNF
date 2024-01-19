import numpy as np

from .utils import triangle_mf, positive_linear_mf, negative_linear_mf, inverse_negative_linear_mf, inverse_positive_linear_mf, inverse_triangle_mf

# ========= PARAMETERS ===========
MF_PARAM_INPUT = [np.array([0.1, 0.5]), 
                  np.array([0.1, 0.5, 0.9]), 
                  np.array([0.5, 0.9])]
MF_PARAM_OUTPUT = [np.array([0.05, 0.2]), 
                   np.array([0.05, 0.2, 0.35]),
                   np.array([0.2, 0.35, 0.5]),
                   np.array([0.35, 0.5, 0.65]),
                   np.array([0.5, 0.65, 0.8]),
                   np.array([0.65, 0.8, 0.95]),
                   np.array([0.8, 0.95])]
MF_FUNCTION_INPUT = [negative_linear_mf, triangle_mf, positive_linear_mf]
MF_FUNCTION_OUTPUT = [inverse_negative_linear_mf, inverse_triangle_mf, inverse_triangle_mf, inverse_triangle_mf, inverse_triangle_mf, inverse_triangle_mf, inverse_positive_linear_mf]
NUM_MF_INPUT = len(MF_PARAM_INPUT)
NUM_MF_OUTPUT = len(MF_PARAM_OUTPUT)

# =========== FUNCTIONS ============
# input -> mf_input
def fuzzify(input):
    mf_input = []
    for i in range(NUM_MF_INPUT):
        mf_input.append(MF_FUNCTION_INPUT[i](*(input, MF_PARAM_INPUT[i])))
    return mf_input

# mf_output -> output
def defuzzify(mf_output):
    # based on the equation in Fuzzy - GA
    centers = []
    for i in range(NUM_MF_OUTPUT):
        centers.append(MF_FUNCTION_OUTPUT[i](*(mf_output[i], MF_PARAM_OUTPUT[i])))
    centers = np.array(centers)
    if isinstance(mf_output, list):
        mf_output = np.array(mf_output)
    
    b = np.sum(mf_output)
    a = np.sum(mf_output*centers)
    return a/b

def fuzzy_infer(mf_input, rules):
    mf_output = [0]*NUM_MF_OUTPUT # contain max mf_output for each mf_unit_output
    
    for i0 in range(NUM_MF_INPUT):
        for i1 in range(NUM_MF_INPUT):
            for i2 in range(NUM_MF_INPUT):
                for i3 in range(NUM_MF_INPUT):
                    for i4 in range(NUM_MF_INPUT):
                        # for one rule
                        mf_one_rule = min(mf_input[i0][0], mf_input[i1][1], mf_input[i2][2], mf_input[i3][3], mf_input[i4][4])
                        mf_unit_output = rules[i0][i1][i2][i3][i4]
                        mf_output[mf_unit_output] = max(mf_output[mf_unit_output],mf_one_rule)  
    return mf_output        
                    
# the whole FIS
def fuzzy_system(input, rules):
    """
        input: array 1D, len = num_features = 5
        rules: array 5D 3x3x3x3x3
    """
    # fuzzification
    mf_input = fuzzify(input) # NUM_MF_INPUT x 5

    # fuzzy infer
    mf_output = fuzzy_infer(mf_input, rules) # NUM_MF_OUTPUT x 1

    # defuzzification
    output = defuzzify(mf_output)
    return output

if __name__ == "__main__":
    input = np.array([0.1,0.3,0.5,0.7,0.9])
    # print(fuzzify(input))

    # create rules
    A = [0.5,0.2,0.3,0.3,0.1]
    B = [1,2,3,3,1]
    rules = np.zeros((NUM_MF_INPUT,)*5) # 5 is dim of input
    for i0 in range(NUM_MF_INPUT):
        for i1 in range(NUM_MF_INPUT):
            for i2 in range(NUM_MF_INPUT):
                for i3 in range(NUM_MF_INPUT):
                    for i4 in range(NUM_MF_INPUT):
                            rules[i0][i1][i2][i3][i4] = A[0]*(i0)**B[0] + A[1]*(i1)**B[1] + A[2]*(i2)**B[2] + A[3]*(i3)**B[3] + A[4]*(i4)**B[4]
    rules = np.ceil(rules / np.max(rules) * 7) - 1 # in range [0,6]
    rules = rules.astype('int32')