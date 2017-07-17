state_infos = [ 
'control blue','control green','control magenta','control red','control teal','control yellow','step 1','step 2','step 3','step 4','step 5','step 6','step 7','step 8','step 9','step 10','step 11','step 12','step 13','step 14','step 15','step 16','step 17','step 18','step 19','step 20','step 21','step 22','step 23','step 24','step 25','step 26','step 27','step 28','step 29','step 30','step 31','step 32','step 33','step 34','step 35','step 36','step 37','step 38','step 39','step 40','step 41','step 42','step 43','step 44','step 45','step 46','step 47','step 48','step 49','step 50','step 51','step 52','step 53','step 54','step 55','step 56','step 57','step 58','step 59','step 60','cell a1 blank','cell a1 red','cell a1 teal','cell b1 blank','cell b1 red','cell b1 teal','cell b2 blank','cell b2 red','cell b2 teal','cell c1 blank','cell c1 green','cell c1 magenta','cell c2 blank','cell c2 green','cell c2 magenta','cell c3 blank','cell c3 blue','cell c3 green','cell c3 magenta','cell c3 red','cell c3 teal','cell c3 yellow','cell c4 blank','cell c4 blue','cell c4 green','cell c4 magenta','cell c4 red','cell c4 teal','cell c4 yellow','cell c5 blank','cell c5 blue','cell c5 green','cell c5 magenta','cell c5 red','cell c5 teal','cell c5 yellow','cell c6 blank','cell c6 blue','cell c6 yellow','cell c7 blank','cell c7 blue','cell c7 yellow','cell d1 blank','cell d1 green','cell d1 magenta','cell d2 blank','cell d2 blue','cell d2 green','cell d2 magenta','cell d2 red','cell d2 teal','cell d2 yellow','cell d3 blank','cell d3 blue','cell d3 green','cell d3 magenta','cell d3 red','cell d3 teal','cell d3 yellow','cell d4 blank','cell d4 blue','cell d4 green','cell d4 magenta','cell d4 red','cell d4 teal','cell d4 yellow','cell d5 blank','cell d5 blue','cell d5 green','cell d5 magenta','cell d5 red','cell d5 teal','cell d5 yellow','cell d6 blank','cell d6 blue','cell d6 yellow','cell e1 blank','cell e1 blue','cell e1 green','cell e1 magenta','cell e1 red','cell e1 teal','cell e1 yellow','cell e2 blank','cell e2 blue','cell e2 green','cell e2 magenta','cell e2 red','cell e2 teal','cell e2 yellow','cell e3 blank','cell e3 blue','cell e3 green','cell e3 magenta','cell e3 red','cell e3 teal','cell e3 yellow','cell e4 blank','cell e4 blue','cell e4 green','cell e4 magenta','cell e4 red','cell e4 teal','cell e4 yellow','cell e5 blank','cell e5 blue','cell e5 green','cell e5 magenta','cell e5 red','cell e5 teal','cell e5 yellow','cell f1 blank','cell f1 blue','cell f1 yellow','cell f2 blank','cell f2 blue','cell f2 green','cell f2 magenta','cell f2 red','cell f2 teal','cell f2 yellow','cell f3 blank','cell f3 blue','cell f3 green','cell f3 magenta','cell f3 red','cell f3 teal','cell f3 yellow','cell f4 blank','cell f4 blue','cell f4 green','cell f4 magenta','cell f4 red','cell f4 teal','cell f4 yellow','cell f5 blank','cell f5 blue','cell f5 green','cell f5 magenta','cell f5 red','cell f5 teal','cell f5 yellow','cell f6 blank','cell f6 green','cell f6 magenta','cell g1 blank','cell g1 blue','cell g1 yellow','cell g2 blank','cell g2 blue','cell g2 yellow','cell g3 blank','cell g3 blue','cell g3 green','cell g3 magenta','cell g3 red','cell g3 teal','cell g3 yellow','cell g4 blank','cell g4 blue','cell g4 green','cell g4 magenta','cell g4 red','cell g4 teal','cell g4 yellow','cell g5 blank','cell g5 blue','cell g5 green','cell g5 magenta','cell g5 red','cell g5 teal','cell g5 yellow','cell g6 blank','cell g6 green','cell g6 magenta','cell g7 blank','cell g7 green','cell g7 magenta','cell h1 blank','cell h1 red','cell h1 teal','cell h2 blank','cell h2 red','cell h2 teal','cell i1 blank','cell i1 red','cell i1 teal'] 

role_0 =['noop','move a1 b1','move a1 b2','move a1 c3','move a1 c5','move a1 e1','move a1 e3','move a1 e5','move b1 c3','move b1 c4','move b1 d2','move b1 d4','move b1 f3','move b1 f5','move b2 c4','move b2 c5','move b2 d3','move b2 d5','move b2 f2','move b2 f4','move c3 d2','move c3 d3','move c3 e1','move c3 e3','move c3 g3','move c3 g5','move c4 d3','move c4 d4','move c4 e2','move c4 e4','move c4 g4','move c5 d4','move c5 d5','move c5 e3','move c5 e5','move c5 g3','move c5 g5','move d2 e1','move d2 e2','move d2 f3','move d2 h2','move d3 e2','move d3 e3','move d3 f2','move d3 f4','move d3 h1','move d4 e3','move d4 e4','move d4 f3','move d4 f5','move d4 h2','move d5 e4','move d5 e5','move d5 f4','move d5 h1','move e1 f2','move e1 g3','move e1 i1','move e2 f2','move e2 f3','move e2 g4','move e3 f3','move e3 f4','move e3 g3','move e3 g5','move e3 i1','move e4 f4','move e4 f5','move e4 g4','move e5 f5','move e5 g5','move e5 i1','move f2 g3','move f2 h1','move f3 g3','move f3 g4','move f3 h2','move f4 g4','move f4 g5','move f4 h1','move f5 g5','move f5 h2','move g3 h1','move g3 i1','move g4 h1','move g4 h2','move g5 h2','move g5 i1','move h1 i1','move h2 i1']
role_1 =['noop','move c3 d2','move c3 e1','move c3 g1','move c4 c3','move c4 d3','move c4 e2','move c4 g2','move c5 c3','move c5 c4','move c5 d4','move c5 e1','move c5 e3','move c5 g3','move c6 c4','move c6 c5','move c6 d5','move c6 e2','move c6 e4','move c6 g4','move c7 c3','move c7 c5','move c7 c6','move c7 d6','move c7 e3','move c7 e5','move c7 g5','move d2 e1','move d2 f1','move d3 d2','move d3 e2','move d3 f2','move d4 d2','move d4 d3','move d4 e3','move d4 f1','move d4 f3','move d5 d3','move d5 d4','move d5 e4','move d5 f2','move d5 f4','move d6 d2','move d6 d4','move d6 d5','move d6 e5','move d6 f3','move d6 f5','move e1 f1','move e1 g1','move e2 e1','move e2 f2','move e2 g2','move e3 e1','move e3 e2','move e3 f3','move e3 g1','move e3 g3','move e4 e2','move e4 e3','move e4 f4','move e4 g2','move e4 g4','move e5 e1','move e5 e3','move e5 e4','move e5 f5','move e5 g3','move e5 g5','move f1 g1','move f2 f1','move f2 g2','move f3 f1','move f3 f2','move f3 g3','move f4 f2','move f4 f3','move f4 g4','move f5 f1','move f5 f3','move f5 f4','move f5 g5','move g2 g1','move g3 g1','move g3 g2','move g4 g2','move g4 g3','move g5 g1','move g5 g3','move g5 g4']
role_2 =['noop','move c2 c1','move c3 c1','move c3 c2','move c4 c2','move c4 c3','move c5 c1','move c5 c3','move c5 c4','move d1 c1','move d2 c2','move d2 d1','move d3 c3','move d3 d1','move d3 d2','move d4 c4','move d4 d2','move d4 d3','move d5 c5','move d5 d1','move d5 d3','move d5 d4','move e1 c1','move e1 d1','move e2 c2','move e2 d2','move e2 e1','move e3 c1','move e3 c3','move e3 d3','move e3 e1','move e3 e2','move e4 c2','move e4 c4','move e4 d4','move e4 e2','move e4 e3','move e5 c3','move e5 c5','move e5 d5','move e5 e1','move e5 e3','move e5 e4','move f2 d1','move f2 e1','move f3 d2','move f3 e2','move f3 f2','move f4 d1','move f4 d3','move f4 e3','move f4 f2','move f4 f3','move f5 d2','move f5 d4','move f5 e4','move f5 f3','move f5 f4','move f6 d3','move f6 d5','move f6 e5','move f6 f2','move f6 f4','move f6 f5','move g3 c1','move g3 e1','move g3 f2','move g4 c2','move g4 e2','move g4 f3','move g4 g3','move g5 c3','move g5 e1','move g5 e3','move g5 f4','move g5 g3','move g5 g4','move g6 c4','move g6 e2','move g6 e4','move g6 f5','move g6 g4','move g6 g5','move g7 c5','move g7 e3','move g7 e5','move g7 f6','move g7 g3','move g7 g5','move g7 g6']
role_3 =['noop','move b1 a1','move b2 a1','move c3 a1','move c3 b1','move c4 b1','move c4 b2','move c5 a1','move c5 b2','move d2 b1','move d2 c3','move d3 b2','move d3 c3','move d3 c4','move d4 b1','move d4 c4','move d4 c5','move d5 b2','move d5 c5','move e1 a1','move e1 c3','move e1 d2','move e2 c4','move e2 d2','move e2 d3','move e3 a1','move e3 c3','move e3 c5','move e3 d3','move e3 d4','move e4 c4','move e4 d4','move e4 d5','move e5 a1','move e5 c5','move e5 d5','move f2 b2','move f2 d3','move f2 e1','move f2 e2','move f3 b1','move f3 d2','move f3 d4','move f3 e2','move f3 e3','move f4 b2','move f4 d3','move f4 d5','move f4 e3','move f4 e4','move f5 b1','move f5 d4','move f5 e4','move f5 e5','move g3 c3','move g3 c5','move g3 e1','move g3 e3','move g3 f2','move g3 f3','move g4 c4','move g4 e2','move g4 e4','move g4 f3','move g4 f4','move g5 c3','move g5 c5','move g5 e3','move g5 e5','move g5 f4','move g5 f5','move h1 d3','move h1 d5','move h1 f2','move h1 f4','move h1 g3','move h1 g4','move h2 d2','move h2 d4','move h2 f3','move h2 f5','move h2 g4','move h2 g5','move i1 e1','move i1 e3','move i1 e5','move i1 g3','move i1 g5','move i1 h1','move i1 h2']
role_4 =['noop','move c3 c4','move c3 c5','move c3 c7','move c4 c5','move c4 c6','move c5 c6','move c5 c7','move c6 c7','move d2 c3','move d2 d3','move d2 d4','move d2 d6','move d3 c4','move d3 d4','move d3 d5','move d4 c5','move d4 d5','move d4 d6','move d5 c6','move d5 d6','move d6 c7','move e1 c3','move e1 c5','move e1 d2','move e1 e2','move e1 e3','move e1 e5','move e2 c4','move e2 c6','move e2 d3','move e2 e3','move e2 e4','move e3 c5','move e3 c7','move e3 d4','move e3 e4','move e3 e5','move e4 c6','move e4 d5','move e4 e5','move e5 c7','move e5 d6','move f1 d2','move f1 d4','move f1 e1','move f1 f2','move f1 f3','move f1 f5','move f2 d3','move f2 d5','move f2 e2','move f2 f3','move f2 f4','move f3 d4','move f3 d6','move f3 e3','move f3 f4','move f3 f5','move f4 d5','move f4 e4','move f4 f5','move f5 d6','move f5 e5','move g1 c3','move g1 e1','move g1 e3','move g1 f1','move g1 g2','move g1 g3','move g1 g5','move g2 c4','move g2 e2','move g2 e4','move g2 f2','move g2 g3','move g2 g4','move g3 c5','move g3 e3','move g3 e5','move g3 f3','move g3 g4','move g3 g5','move g4 c6','move g4 e4','move g4 f4','move g4 g5','move g5 c7','move g5 e5','move g5 f5']
role_5 =['noop','move c1 c2','move c1 c3','move c1 c5','move c1 d1','move c1 e1','move c1 e3','move c1 g3','move c2 c3','move c2 c4','move c2 d2','move c2 e2','move c2 e4','move c2 g4','move c3 c4','move c3 c5','move c3 d3','move c3 e3','move c3 e5','move c3 g5','move c4 c5','move c4 d4','move c4 e4','move c4 g6','move c5 d5','move c5 e5','move c5 g7','move d1 d2','move d1 d3','move d1 d5','move d1 e1','move d1 f2','move d1 f4','move d2 d3','move d2 d4','move d2 e2','move d2 f3','move d2 f5','move d3 d4','move d3 d5','move d3 e3','move d3 f4','move d3 f6','move d4 d5','move d4 e4','move d4 f5','move d5 e5','move d5 f6','move e1 e2','move e1 e3','move e1 e5','move e1 f2','move e1 g3','move e1 g5','move e2 e3','move e2 e4','move e2 f3','move e2 g4','move e2 g6','move e3 e4','move e3 e5','move e3 f4','move e3 g5','move e3 g7','move e4 e5','move e4 f5','move e4 g6','move e5 f6','move e5 g7','move f2 f3','move f2 f4','move f2 f6','move f2 g3','move f3 f4','move f3 f5','move f3 g4','move f4 f5','move f4 f6','move f4 g5','move f5 f6','move f5 g6','move f6 g7','move g3 g4','move g3 g5','move g3 g7','move g4 g5','move g4 g6','move g5 g6','move g5 g7','move g6 g7']

roles = [role_0, role_1, role_2, role_3, role_4, role_5]

files = [
    open("chinese_checkers_6_role0.csv","a"),
    open("chinese_checkers_6_role1.csv","a"),
    open("chinese_checkers_6_role2.csv","a"),
    open("chinese_checkers_6_role3.csv","a"),
    open("chinese_checkers_6_role4.csv","a"),
    open("chinese_checkers_6_role5.csv","a")
]

with open("chinesecheckers6.log","r") as infile:
    while True:
        state_str = infile.readline()
        if not state_str: break  # EOF
        state_csv_str = ""
        for info in state_infos:
            state_csv_str += ("1," if info in state_str else "-1,")

        action_str = infile.readline()
        
        roleId = 0
        action_csv_str = "" # Never include noop (the first action) 
        for info in action_str.split(','):
            if 'noop' not in info:
                for action in roles[roleId]:
                    action_csv_str += ("1," if action in info else "0,")
                break

            roleId += 1
        if roleId == 6:
            continue
            
        header = "extra_data,-1,%d,100," % roleId
        line = header + state_csv_str + action_csv_str.strip(",") + "\n"
        files[roleId].write(line)

for f in files:
    f.close()


#STATE: ([( true ( cell a1 red ) ), ( true ( cell b1 red ) ), ( true ( cell b2 red ) ), ( true ( cell c1 magenta ) ), ( true ( cell c2 magenta ) ), ( true ( cell c3 blank ) ), ( true ( cell c4 blank ) ), ( true ( cell c5 blank ) ), ( true ( cell c6 yellow ) ), ( true ( cell c7 yellow ) ), ( true ( cell d1 magenta ) ), ( true ( cell d2 blank ) ), ( true ( cell d3 blank ) ), ( true ( cell d4 blank ) ), ( true ( cell d5 blank ) ), ( true ( cell d6 yellow ) ), ( true ( cell e1 blank ) ), ( true ( cell e2 blank ) ), ( true ( cell e3 blank ) ), ( true ( cell e4 blank ) ), ( true ( cell e5 blank ) ), ( true ( cell f1 blue ) ), ( true ( cell f2 blank ) ), ( true ( cell f3 blank ) ), ( true ( cell f4 blank ) ), ( true ( cell f5 blank ) ), ( true ( cell f6 green ) ), ( true ( cell g1 blue ) ), ( true ( cell g2 blue ) ), ( true ( cell g3 blank ) ), ( true ( cell g4 blank ) ), ( true ( cell g5 blank ) ), ( true ( cell g6 green ) ), ( true ( cell g7 green ) ), ( true ( cell h1 teal ) ), ( true ( cell h2 teal ) ), ( true ( cell i1 teal ) ), ( true ( control red ) ), ( true ( step 1 ) )])

#MOVE: (does red ( move a1 c3 )), (does yellow noop), (does green noop), (does teal noop), (does blue noop), (does magenta noop)