from kaggle_environments import make

env = make("mab", debug=True)

env.reset()
env.run(["submission.py", "ucb_decay.py"])
env.render(mode="ipython", width=800, height=500)
'''
env.reset()
env.run(["../input/santa-2020/submission.py", "bayesian_ucb.py"])
env.render(mode="ipython", width=800, height=500)

def print_rounds(file1, file2, N=5):
    env = make("mab", debug=True)

    for i in range(N):
        env.run([file1, file2])
        p1_score = env.steps[-1][0]['reward']
        p2_score = env.steps[-1][1]['reward']
        env.reset()
        print(f"Round {i+1}: {p1_score} - {p2_score}")



print('Default vs UCB+decay')
print_rounds("../input/santa-2020/submission.py", "ucb_decay.py")



print('Default vs BayesianUCB')
print_rounds("../input/santa-2020/submission.py", "bayesian_ucb.py")
'''

