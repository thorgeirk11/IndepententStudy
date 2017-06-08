select 
	m.match_id ,
	s.step_number,
    p.roleindex,
    p.goal_value,
    
    CONTROLRED, CONTROLWHITE,
    CELL10DIRT,
    CELL11B, CELL11R, CELL11W, CELL12B, CELL12R, CELL12W,
    CELL13B, CELL13R, CELL13W, CELL14B, CELL14R, CELL14W,
    CELL15B, CELL15R, CELL15W, CELL16B, CELL16R, CELL16W,
    CELL20DIRT,
    CELL21B, CELL21R, CELL21W, CELL22B, CELL22R, CELL22W,
    CELL23B, CELL23R, CELL23W, CELL24B, CELL24R, CELL24W,
    CELL25B, CELL25R, CELL25W, CELL26B, CELL26R, CELL26W,
    CELL30DIRT,
    CELL31B, CELL31R, CELL31W, CELL32B, CELL32R, CELL32W,
    CELL33B, CELL33R, CELL33W, CELL34B, CELL34R, CELL34W,
    CELL35B, CELL35R, CELL35W, CELL36B, CELL36R, CELL36W,
    CELL40DIRT,
    CELL41B, CELL41R, CELL41W, CELL42B, CELL42R, CELL42W,
    CELL43B, CELL43R, CELL43W, CELL44B, CELL44R, CELL44W,
    CELL45B, CELL45R, CELL45W, CELL46B, CELL46R, CELL46W,
    CELL50DIRT,
    CELL51B, CELL51R, CELL51W, CELL52B, CELL52R, CELL52W,
    CELL53B, CELL53R, CELL53W, CELL54B, CELL54R, CELL54W,
    CELL55B, CELL55R, CELL55W, CELL56B, CELL56R, CELL56W,
    CELL60DIRT,
    CELL61B, CELL61R, CELL61W, CELL62B, CELL62R, CELL62W,
    CELL63B, CELL63R, CELL63W, CELL64B, CELL64R, CELL64W,
    CELL65B, CELL65R, CELL65W, CELL66B, CELL66R, CELL66W,
    CELL70DIRT,
    CELL71B, CELL71R, CELL71W, CELL72B, CELL72R, CELL72W,
    CELL73B, CELL73R, CELL73W, CELL74B, CELL74R, CELL74W,
    CELL75B, CELL75R, CELL75W, CELL76B, CELL76R, CELL76W,
    
    locate('NOOP', move)     as NOOP,
    locate('(DROP 1)', move) as DROP1,
    locate('(DROP 2)', move) as DROP2,
    locate('(DROP 3)', move) as DROP3,
    locate('(DROP 4)', move) as DROP4,
    locate('(DROP 5)', move) as DROP5,
    locate('(DROP 6)', move) as DROP6,
    locate('(DROP 7)', move) as DROP7
FROM 
    states_parsed s, 
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

    m.game = 'connect4' and 
    p.player != 'Random'