select 
	s.state,
	m.match_id ,
	s.step_number,
    p.roleindex,
    p.goal_value,
    mo.move
FROM 
    states s, 
    matches m, 
    match_players p,
    moves mo
where 
    m.match_id = s.match_id and
    p.match_id = m.match_id and
    p.match_id = m.match_id and
    p.roleindex = mo.roleindex and
    mo.roleindex = p.roleindex and
    mo.match_id = m.match_id and
    mo.step_number = s.step_number and

    m.game = 'chinesecheckers6' and 
    p.player != 'Random'