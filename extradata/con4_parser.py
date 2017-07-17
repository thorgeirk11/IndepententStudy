
state_infos = ['control blue','control green','cell 1 1 b','cell 1 1 r','cell 1 1 w','cell 1 2 b','cell 1 2 r','cell 1 2 w','cell 1 3 b','cell 1 3 r','cell 1 3 w','cell 1 4 b','cell 1 4 r','cell 1 4 w','cell 1 5 b','cell 1 5 r','cell 1 5 w','cell 1 6 b','cell 1 6 r','cell 1 6 w','cell 2 1 b','cell 2 1 r','cell 2 1 w','cell 2 2 b','cell 2 2 r','cell 2 2 w','cell 2 3 b','cell 2 3 r','cell 2 3 w','cell 2 4 b','cell 2 4 r','cell 2 4 w','cell 2 5 b','cell 2 5 r','cell 2 5 w','cell 2 6 b','cell 2 6 r','cell 2 6 w','cell 3 1 b','cell 3 1 r','cell 3 1 w','cell 3 2 b','cell 3 2 r','cell 3 2 w','cell 3 3 b','cell 3 3 r','cell 3 3 w','cell 3 4 b','cell 3 4 r','cell 3 4 w','cell 3 5 b','cell 3 5 r','cell 3 5 w','cell 3 6 b','cell 3 6 r','cell 3 6 w','cell 4 1 b','cell 4 1 r','cell 4 1 w','cell 4 2 b','cell 4 2 r','cell 4 2 w','cell 4 3 b','cell 4 3 r','cell 4 3 w','cell 4 4 b','cell 4 4 r','cell 4 4 w','cell 4 5 b','cell 4 5 r','cell 4 5 w','cell 4 6 b','cell 4 6 r','cell 4 6 w','cell 5 1 b','cell 5 1 r','cell 5 1 w','cell 5 2 b','cell 5 2 r','cell 5 2 w','cell 5 3 b','cell 5 3 r','cell 5 3 w','cell 5 4 b','cell 5 4 r','cell 5 4 w','cell 5 5 b','cell 5 5 r','cell 5 5 w','cell 5 6 b','cell 5 6 r','cell 5 6 w','cell 6 1 b','cell 6 1 r','cell 6 1 w','cell 6 2 b','cell 6 2 r','cell 6 2 w','cell 6 3 b','cell 6 3 r','cell 6 3 w','cell 6 4 b','cell 6 4 r','cell 6 4 w','cell 6 5 b','cell 6 5 r','cell 6 5 w','cell 6 6 b','cell 6 6 r','cell 6 6 w','cell 7 1 b','cell 7 1 r','cell 7 1 w','cell 7 2 b','cell 7 2 r','cell 7 2 w','cell 7 3 b','cell 7 3 r','cell 7 3 w','cell 7 4 b','cell 7 4 r','cell 7 4 w','cell 7 5 b','cell 7 5 r','cell 7 5 w','cell 7 6 b','cell 7 6 r','cell 7 6 w']

actions = ['noop','drop 1','drop 2','drop 3','drop 4','drop 5','drop 6','drop 7']
files = [open("connect4_role0.csv","a"), open("connect4_role1.csv","a")]

with open("connect4.log","r") as infile:
    while True:
        state_str = infile.readline()
        if not state_str: break  # EOF
        state_csv_str = ""
        for info in state_infos:
            state_csv_str += ("1," if info in state_str else "-1,")

        action_str = infile.readline()
        
        roleId = 0
        action_csv_str = "" 
        for info in action_str.split(','):
            if 'noop' not in info:
                for action in actions:
                    action_csv_str += ("1," if action in info else "0,")
                break
            roleId += 1
            
        header = "extra_data,-1,%d,100," % roleId
        line = header + state_csv_str + action_csv_str.strip(",") + "\n"
        files[roleId].write(line)

for f in files:
    f.close()