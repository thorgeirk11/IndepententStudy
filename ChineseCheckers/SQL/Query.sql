select g.name, count(m.match_id) MatchCount, sum(temp.step) as StepCount  from games g
inner join matches m on m.game = g.name
inner join 
	(
		select match_id, max(step_number) as step 
		from moves 
		group by match_id
	) temp on temp.match_id = m.match_id
where m.match_id not in (select match_id from errormessages)
group by g.name
