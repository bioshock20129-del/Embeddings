from pyan import create_callgraph

options = {'colored': True, 'grouped': True}

output = create_callgraph("agent.py", **options)

with open('graph.dot', 'w') as f:
    f.write(output)
