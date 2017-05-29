
drop table states_parsed;
create table states_parsed(
	game varchar(40),
	match_id varchar(40),
    step_number int(11),
	CONTROLRED int unsigned,
	CONTROLWHITE int unsigned,
	CELL10DIRT int unsigned,
	CELL11B int unsigned,
	CELL11R int unsigned,
	CELL11W int unsigned,
	CELL12B int unsigned,
	CELL12R int unsigned,
	CELL12W int unsigned,
	CELL13B int unsigned,
	CELL13R int unsigned,
	CELL13W int unsigned,
	CELL14B int unsigned,
	CELL14R int unsigned,
	CELL14W int unsigned,
	CELL15B int unsigned,
	CELL15R int unsigned,
	CELL15W int unsigned,
	CELL16B int unsigned,
	CELL16R int unsigned,
	CELL16W int unsigned,
	CELL20DIRT int unsigned,
	CELL21B int unsigned,
	CELL21R int unsigned,
	CELL21W int unsigned,
	CELL22B int unsigned,
	CELL22R int unsigned,
	CELL22W int unsigned,
	CELL23B int unsigned,
	CELL23R int unsigned,
	CELL23W int unsigned,
	CELL24B int unsigned,
	CELL24R int unsigned,
	CELL24W int unsigned,
	CELL25B int unsigned,
	CELL25R int unsigned,
	CELL25W int unsigned,
	CELL26B int unsigned,
	CELL26R int unsigned,
	CELL26W int unsigned,
	CELL30DIRT int unsigned,
	CELL31B int unsigned,
	CELL31R int unsigned,
	CELL31W int unsigned,
	CELL32B int unsigned,
	CELL32R int unsigned,
	CELL32W int unsigned,
	CELL33B int unsigned,
	CELL33R int unsigned,
	CELL33W int unsigned,
	CELL34B int unsigned,
	CELL34R int unsigned,
	CELL34W int unsigned,
	CELL35B int unsigned,
	CELL35R int unsigned,
	CELL35W int unsigned,
	CELL36B int unsigned,
	CELL36R int unsigned,
	CELL36W int unsigned,
	CELL40DIRT int unsigned,
	CELL41B int unsigned,
	CELL41R int unsigned,
	CELL41W int unsigned,
	CELL42B int unsigned,
	CELL42R int unsigned,
	CELL42W int unsigned,
	CELL43B int unsigned,
	CELL43R int unsigned,
	CELL43W int unsigned,
	CELL44B int unsigned,
	CELL44R int unsigned,
	CELL44W int unsigned,
	CELL45B int unsigned,
	CELL45R int unsigned,
	CELL45W int unsigned,
	CELL46B int unsigned,
	CELL46R int unsigned,
	CELL46W int unsigned,
	CELL50DIRT int unsigned,
	CELL51B int unsigned,
	CELL51R int unsigned,
	CELL51W int unsigned,
	CELL52B int unsigned,
	CELL52R int unsigned,
	CELL52W int unsigned,
	CELL53B int unsigned,
	CELL53R int unsigned,
	CELL53W int unsigned,
	CELL54B int unsigned,
	CELL54R int unsigned,
	CELL54W int unsigned,
	CELL55B int unsigned,
	CELL55R int unsigned,
	CELL55W int unsigned,
	CELL56B int unsigned,
	CELL56R int unsigned,
	CELL56W int unsigned,
	CELL60DIRT int unsigned,
	CELL61B int unsigned,
	CELL61R int unsigned,
	CELL61W int unsigned,
	CELL62B int unsigned,
	CELL62R int unsigned,
	CELL62W int unsigned,
	CELL63B int unsigned,
	CELL63R int unsigned,
	CELL63W int unsigned,
	CELL64B int unsigned,
	CELL64R int unsigned,
	CELL64W int unsigned,
	CELL65B int unsigned,
	CELL65R int unsigned,
	CELL65W int unsigned,
	CELL66B int unsigned,
	CELL66R int unsigned,
	CELL66W int unsigned,
	CELL70DIRT int unsigned,
	CELL71B int unsigned,
	CELL71R int unsigned,
	CELL71W int unsigned,
	CELL72B int unsigned,
	CELL72R int unsigned,
	CELL72W int unsigned,
	CELL73B int unsigned,
	CELL73R int unsigned,
	CELL73W int unsigned,
	CELL74B int unsigned,
	CELL74R int unsigned,
	CELL74W int unsigned,
	CELL75B int unsigned,
	CELL75R int unsigned,
	CELL75W int unsigned,
	CELL76B int unsigned,
	CELL76R int unsigned,
	CELL76W int unsigned
);
insert into states_parsed 
(select 
	game,
    m.match_id,
    s.step_number,
	CAST(LOCATE('CONTROL RED',state) > 0 as unsigned) as   CONTROLRED,
	CAST(LOCATE('CONTROL WHITE',state) > 0 as unsigned) as CONTROLWHITE,
	CAST(LOCATE('CELL 1 0 DIRT',state) > 0 as unsigned) as CELL10DIRT,
	CAST(LOCATE('CELL 1 1 B',state) > 0 as unsigned) as 	 CELL11B,
	CAST(LOCATE('CELL 1 1 R',state) > 0 as unsigned) as 	 CELL11R,
	CAST(LOCATE('CELL 1 1 W',state) > 0 as unsigned) as 	 CELL11W,
	CAST(LOCATE('CELL 1 2 B',state) > 0 as unsigned) as 	 CELL12B,
	CAST(LOCATE('CELL 1 2 R',state) > 0 as unsigned) as 	 CELL12R,
	CAST(LOCATE('CELL 1 2 W',state) > 0 as unsigned) as 	 CELL12W,
	CAST(LOCATE('CELL 1 3 B',state) > 0 as unsigned) as 	 CELL13B,
	CAST(LOCATE('CELL 1 3 R',state) > 0 as unsigned) as 	 CELL13R,
	CAST(LOCATE('CELL 1 3 W',state) > 0 as unsigned) as 	 CELL13W,
	CAST(LOCATE('CELL 1 4 B',state) > 0 as unsigned) as 	 CELL14B,
	CAST(LOCATE('CELL 1 4 R',state) > 0 as unsigned) as 	 CELL14R,
	CAST(LOCATE('CELL 1 4 W',state) > 0 as unsigned) as 	 CELL14W,
	CAST(LOCATE('CELL 1 5 B',state) > 0 as unsigned) as 	 CELL15B,
	CAST(LOCATE('CELL 1 5 R',state) > 0 as unsigned) as 	 CELL15R,
	CAST(LOCATE('CELL 1 5 W',state) > 0 as unsigned) as 	 CELL15W,
	CAST(LOCATE('CELL 1 6 B',state) > 0 as unsigned) as 	 CELL16B,
	CAST(LOCATE('CELL 1 6 R',state) > 0 as unsigned) as 	 CELL16R,
	CAST(LOCATE('CELL 1 6 W',state) > 0 as unsigned) as 	 CELL16W,
	CAST(LOCATE('CELL 2 0 DIRT',state) > 0 as unsigned) as CELL20DIRT,
	CAST(LOCATE('CELL 2 1 B',state) > 0 as unsigned) as    CELL21B,
	CAST(LOCATE('CELL 2 1 R',state) > 0 as unsigned) as    CELL21R,
	CAST(LOCATE('CELL 2 1 W',state) > 0 as unsigned) as    CELL21W,
	CAST(LOCATE('CELL 2 2 B',state) > 0 as unsigned) as    CELL22B,
	CAST(LOCATE('CELL 2 2 R',state) > 0 as unsigned) as    CELL22R,
	CAST(LOCATE('CELL 2 2 W',state) > 0 as unsigned) as    CELL22W,
	CAST(LOCATE('CELL 2 3 B',state) > 0 as unsigned) as    CELL23B,
	CAST(LOCATE('CELL 2 3 R',state) > 0 as unsigned) as    CELL23R,
	CAST(LOCATE('CELL 2 3 W',state) > 0 as unsigned) as    CELL23W,
	CAST(LOCATE('CELL 2 4 B',state) > 0 as unsigned) as    CELL24B,
	CAST(LOCATE('CELL 2 4 R',state) > 0 as unsigned) as    CELL24R,
	CAST(LOCATE('CELL 2 4 W',state) > 0 as unsigned) as    CELL24W,
	CAST(LOCATE('CELL 2 5 B',state) > 0 as unsigned) as    CELL25B,
	CAST(LOCATE('CELL 2 5 R',state) > 0 as unsigned) as    CELL25R,
	CAST(LOCATE('CELL 2 5 W',state) > 0 as unsigned) as    CELL25W,
	CAST(LOCATE('CELL 2 6 B',state) > 0 as unsigned) as    CELL26B,
	CAST(LOCATE('CELL 2 6 R',state) > 0 as unsigned) as    CELL26R,
	CAST(LOCATE('CELL 2 6 W',state) > 0 as unsigned) as    CELL26W,
	CAST(LOCATE('CELL 3 0 DIRT',state) > 0 as unsigned) as CELL30DIRT,
	CAST(LOCATE('CELL 3 1 B',state) > 0 as unsigned) as    CELL31B,
	CAST(LOCATE('CELL 3 1 R',state) > 0 as unsigned) as    CELL31R,
	CAST(LOCATE('CELL 3 1 W',state) > 0 as unsigned) as    CELL31W,
	CAST(LOCATE('CELL 3 2 B',state) > 0 as unsigned) as    CELL32B,
	CAST(LOCATE('CELL 3 2 R',state) > 0 as unsigned) as    CELL32R,
	CAST(LOCATE('CELL 3 2 W',state) > 0 as unsigned) as    CELL32W,
	CAST(LOCATE('CELL 3 3 B',state) > 0 as unsigned) as    CELL33B,
	CAST(LOCATE('CELL 3 3 R',state) > 0 as unsigned) as    CELL33R,
	CAST(LOCATE('CELL 3 3 W',state) > 0 as unsigned) as    CELL33W,
	CAST(LOCATE('CELL 3 4 B',state) > 0 as unsigned) as    CELL34B,
	CAST(LOCATE('CELL 3 4 R',state) > 0 as unsigned) as    CELL34R,
	CAST(LOCATE('CELL 3 4 W',state) > 0 as unsigned) as    CELL34W,
	CAST(LOCATE('CELL 3 5 B',state) > 0 as unsigned) as    CELL35B,
	CAST(LOCATE('CELL 3 5 R',state) > 0 as unsigned) as    CELL35R,
	CAST(LOCATE('CELL 3 5 W',state) > 0 as unsigned) as    CELL35W,
	CAST(LOCATE('CELL 3 6 B',state) > 0 as unsigned) as    CELL36B,
	CAST(LOCATE('CELL 3 6 R',state) > 0 as unsigned) as    CELL36R,
	CAST(LOCATE('CELL 3 6 W',state) > 0 as unsigned) as    CELL36W,
	CAST(LOCATE('CELL 4 0 DIRT',state) > 0 as unsigned) as CELL40DIRT,
	CAST(LOCATE('CELL 4 1 B',state) > 0 as unsigned) as    CELL41B,
	CAST(LOCATE('CELL 4 1 R',state) > 0 as unsigned) as    CELL41R,
	CAST(LOCATE('CELL 4 1 W',state) > 0 as unsigned) as    CELL41W,
	CAST(LOCATE('CELL 4 2 B',state) > 0 as unsigned) as    CELL42B,
	CAST(LOCATE('CELL 4 2 R',state) > 0 as unsigned) as    CELL42R,
	CAST(LOCATE('CELL 4 2 W',state) > 0 as unsigned) as    CELL42W,
	CAST(LOCATE('CELL 4 3 B',state) > 0 as unsigned) as    CELL43B,
	CAST(LOCATE('CELL 4 3 R',state) > 0 as unsigned) as    CELL43R,
	CAST(LOCATE('CELL 4 3 W',state) > 0 as unsigned) as    CELL43W,
	CAST(LOCATE('CELL 4 4 B',state) > 0 as unsigned) as    CELL44B,
	CAST(LOCATE('CELL 4 4 R',state) > 0 as unsigned) as    CELL44R,
	CAST(LOCATE('CELL 4 4 W',state) > 0 as unsigned) as    CELL44W,
	CAST(LOCATE('CELL 4 5 B',state) > 0 as unsigned) as    CELL45B,
	CAST(LOCATE('CELL 4 5 R',state) > 0 as unsigned) as    CELL45R,
	CAST(LOCATE('CELL 4 5 W',state) > 0 as unsigned) as    CELL45W,
	CAST(LOCATE('CELL 4 6 B',state) > 0 as unsigned) as    CELL46B,
	CAST(LOCATE('CELL 4 6 R',state) > 0 as unsigned) as    CELL46R,
	CAST(LOCATE('CELL 4 6 W',state) > 0 as unsigned) as    CELL46W,
	CAST(LOCATE('CELL 5 0 DIRT',state) > 0 as unsigned) as CELL50DIRT,
	CAST(LOCATE('CELL 5 1 B',state) > 0 as unsigned) as 	 CELL51B,
	CAST(LOCATE('CELL 5 1 R',state) > 0 as unsigned) as 	 CELL51R,
	CAST(LOCATE('CELL 5 1 W',state) > 0 as unsigned) as 	 CELL51W,
	CAST(LOCATE('CELL 5 2 B',state) > 0 as unsigned) as 	 CELL52B,
	CAST(LOCATE('CELL 5 2 R',state) > 0 as unsigned) as 	 CELL52R,
	CAST(LOCATE('CELL 5 2 W',state) > 0 as unsigned) as 	 CELL52W,
	CAST(LOCATE('CELL 5 3 B',state) > 0 as unsigned) as 	 CELL53B,
	CAST(LOCATE('CELL 5 3 R',state) > 0 as unsigned) as 	 CELL53R,
	CAST(LOCATE('CELL 5 3 W',state) > 0 as unsigned) as 	 CELL53W,
	CAST(LOCATE('CELL 5 4 B',state) > 0 as unsigned) as 	 CELL54B,
	CAST(LOCATE('CELL 5 4 R',state) > 0 as unsigned) as 	 CELL54R,
	CAST(LOCATE('CELL 5 4 W',state) > 0 as unsigned) as 	 CELL54W,
	CAST(LOCATE('CELL 5 5 B',state) > 0 as unsigned) as 	 CELL55B,
	CAST(LOCATE('CELL 5 5 R',state) > 0 as unsigned) as 	 CELL55R,
	CAST(LOCATE('CELL 5 5 W',state) > 0 as unsigned) as 	 CELL55W,
	CAST(LOCATE('CELL 5 6 B',state) > 0 as unsigned) as 	 CELL56B,
	CAST(LOCATE('CELL 5 6 R',state) > 0 as unsigned) as 	 CELL56R,
	CAST(LOCATE('CELL 5 6 W',state) > 0 as unsigned) as 	 CELL56W,
	CAST(LOCATE('CELL 6 0 DIRT',state) > 0 as unsigned) as CELL60DIRT,
	CAST(LOCATE('CELL 6 1 B',state) > 0 as unsigned) as    CELL61B,
	CAST(LOCATE('CELL 6 1 R',state) > 0 as unsigned) as    CELL61R,
	CAST(LOCATE('CELL 6 1 W',state) > 0 as unsigned) as    CELL61W,
	CAST(LOCATE('CELL 6 2 B',state) > 0 as unsigned) as    CELL62B,
	CAST(LOCATE('CELL 6 2 R',state) > 0 as unsigned) as    CELL62R,
	CAST(LOCATE('CELL 6 2 W',state) > 0 as unsigned) as    CELL62W,
	CAST(LOCATE('CELL 6 3 B',state) > 0 as unsigned) as    CELL63B,
	CAST(LOCATE('CELL 6 3 R',state) > 0 as unsigned) as    CELL63R,
	CAST(LOCATE('CELL 6 3 W',state) > 0 as unsigned) as    CELL63W,
	CAST(LOCATE('CELL 6 4 B',state) > 0 as unsigned) as    CELL64B,
	CAST(LOCATE('CELL 6 4 R',state) > 0 as unsigned) as    CELL64R,
	CAST(LOCATE('CELL 6 4 W',state) > 0 as unsigned) as    CELL64W,
	CAST(LOCATE('CELL 6 5 B',state) > 0 as unsigned) as    CELL65B,
	CAST(LOCATE('CELL 6 5 R',state) > 0 as unsigned) as    CELL65R,
	CAST(LOCATE('CELL 6 5 W',state) > 0 as unsigned) as    CELL65W,
	CAST(LOCATE('CELL 6 6 B',state) > 0 as unsigned) as    CELL66B,
	CAST(LOCATE('CELL 6 6 R',state) > 0 as unsigned) as    CELL66R,
	CAST(LOCATE('CELL 6 6 W',state) > 0 as unsigned) as    CELL66W,
	CAST(LOCATE('CELL 7 0 DIRT',state) > 0 as unsigned) as CELL70DIRT,
	CAST(LOCATE('CELL 7 1 B',state) > 0 as unsigned) as    CELL71B,
	CAST(LOCATE('CELL 7 1 R',state) > 0 as unsigned) as    CELL71R,
	CAST(LOCATE('CELL 7 1 W',state) > 0 as unsigned) as    CELL71W,
	CAST(LOCATE('CELL 7 2 B',state) > 0 as unsigned) as    CELL72B,
	CAST(LOCATE('CELL 7 2 R',state) > 0 as unsigned) as    CELL72R,
	CAST(LOCATE('CELL 7 2 W',state) > 0 as unsigned) as    CELL72W,
	CAST(LOCATE('CELL 7 3 B',state) > 0 as unsigned) as    CELL73B,
	CAST(LOCATE('CELL 7 3 R',state) > 0 as unsigned) as    CELL73R,
	CAST(LOCATE('CELL 7 3 W',state) > 0 as unsigned) as    CELL73W,
	CAST(LOCATE('CELL 7 4 B',state) > 0 as unsigned) as    CELL74B,
	CAST(LOCATE('CELL 7 4 R',state) > 0 as unsigned) as    CELL74R,
	CAST(LOCATE('CELL 7 4 W',state) > 0 as unsigned) as    CELL74W,
	CAST(LOCATE('CELL 7 5 B',state) > 0 as unsigned) as    CELL75B,
	CAST(LOCATE('CELL 7 5 R',state) > 0 as unsigned) as    CELL75R,
	CAST(LOCATE('CELL 7 5 W',state) > 0 as unsigned) as    CELL75W,
	CAST(LOCATE('CELL 7 6 B',state) > 0 as unsigned) as    CELL76B,
	CAST(LOCATE('CELL 7 6 R',state) > 0 as unsigned) as    CELL76R,
	CAST(LOCATE('CELL 7 6 W',state) > 0 as unsigned) as    CELL76W
	FROM states s
	INNER JOIN matches m ON m.match_id = s.match_id
	WHERE
		m.game = 'connect4'
);
