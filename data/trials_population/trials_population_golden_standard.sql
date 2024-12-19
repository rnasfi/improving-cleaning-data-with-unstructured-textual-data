set search_path = "eudract";

with popul as (
    select distinct clinical_trial_id, protocol_country_code, pd.ddcm_source,
                   --healthy or not
    case when e.healthy_volunteers = 'Yes' then 1 when e.healthy_volunteers = 'No' then 0
        else null end as healthy_volunteers,
    -- gender
    case when e.female = 'Yes' then 1  when e.female = 'No' then 0
        else null end as female,
    case when e.male = 'Yes' then 1 when e.male = 'No' then 0
        else null end as male,
    case when e.others = 'Yes' then 1 when e.others = 'No' then 0
        else null end as others,
    -- age
    case when e.elderly = 'Yes' then 1  when e.elderly = 'No' then 0
        else null end as elderly,
    case when e.adults = 'Yes' then 1 when e.adults = 'No' then 0
        else null end as adults,
    case when e.adolescents = 'Yes' then 1 when e.adolescents = 'No' then 0
        else null end as adolescents,
    case when e.children = 'Yes' then 1 when e.children = 'No' then 0
        else null end  as children
    from eudract.trial_protocols tp  join eudract.protocol_detail pd using (trial_protocols_id)
        join eudract.trial_population e using (protocol_detail_id)
), eudract_titles as (
    select distinct title, ct.eudract_number, t.ddcm_source, clinical_trial_id,
                    medical_condition, gender, population_age
    from eudract.clinical_trial ct
    join eudract.clinical_trial_title  t using (clinical_trial_id)
), euctr as (-- get the full population attributes + eudract_number + country code
                select distinct et.eudract_number, protocol_country_code, --
                    healthy_volunteers, elderly,adults, adolescents,
                    children,
                    female, male, others, population_age
                from eudract_titles et
                left join popul using (clinical_trial_id, ddcm_source)
                order by eudract_number
), ctgov as (-- get the population attributes from CTGov + NCT number + eudract_number
    select distinct  'ctgov' as protocol_country_code,
                     ct.nct,
                     -- n.healthy_volunteers as nct_health_ctgov,
                    case when n.healthy_volunteers = 'Accepts Healthy Volunteers'
                        then 1 else 0 end as healthy_volunteers,
                     -- n.gender as gender,
                    case when lower(n.gender) ilike 'all' or lower(n.gender) ilike 'male'
                        then 1  else 0 end as male,
                    case when lower(n.gender) ~  '.*all.*|female.*'
                        then 1  else 0 end as female,
                    case when lower(n.gender) ilike 'all'
                        then 1  else 0 end as others,
                    -- n.maximum_age as maxi, n.minimum_age as mini,
                    case when eudract.check_age(n.maximum_age, n.minimum_age, 65, 300) = 'Yes'
                        then 1 else 0 end as elderly,
                    case when eudract.check_age(n.maximum_age, n.minimum_age, 17, 65) = 'Yes'
                        then 1 else 0 end as adults,
                    case when eudract.check_age(n.maximum_age, n.minimum_age, 10, 17) = 'Yes'
                        then 1 else 0 end as adolescents,
                    case when eudract.check_age(n.maximum_age, n.minimum_age, 1, 10) = 'Yes'
                        then 1 else 0 end as children,
                    p.eudract_number
    from ctgov.eligibility n
    join ctgov.clinical_trial ct using (clinical_trial_id)
    right join ctgov.eudract_number p using (clinical_trial_id)
    where p.eudract_number is not null
    )
    ,
drks as (
    select distinct 'drks' as protocol_country_code, ct.nct_number as nct,
                    case when lower(n.inclusion_gender) ilike 'Both, male and female' or lower(n.inclusion_gender) ilike 'male'
                        then 1  else 0 end as male,
                    case when lower(n.inclusion_gender) ~  '.*Both, male and female.*|female.*'
                        then 1  else 0 end as female,
                    case when lower(n.inclusion_gender) ilike 'Both, male and female'
                        then 1  else 0 end as others,
                    case when eudract.check_age(n.inclusion_maximum_age, n.inclusion_minimum_age, 65, 300) = 'Yes'
                        then 1 else 0 end as elderly,
                    case when eudract.check_age(n.inclusion_maximum_age, n.inclusion_minimum_age, 17, 65) = 'Yes'
                        then 1 else 0 end as adults,
                    case when eudract.check_age(n.inclusion_maximum_age, n.inclusion_minimum_age, 10, 17) = 'Yes'
                        then 1 else 0 end as adolescents,
                    case when eudract.check_age(n.inclusion_maximum_age, n.inclusion_minimum_age, 1, 10) = 'Yes'
                        then 1 else 0 end as children,
                                    p.eudract_number
        from german_trials.clinical_trial n
        join german_trials.eudract_number p using (clinical_trial_id)
        join german_trials.nct_number ct using (clinical_trial_id)
        where eudract_number is not null
    ),
    euctr_population as (
    select eudract_number, protocol_country_code as country,
           elderly, adults,adolescents, children,
           female, male, others, healthy_volunteers
    from euctr
    order by eudract_number
)
, gs_eudract as(
    select distinct el.eudract_number,
        hl.healthy_volunteers, hy, hn,
        el.elderly, ely, eln,
        ad.adults, ady, adn,
        ado.adolescents, adoy, adon,
        ch.children, chy, chn,
        f.female, fy, fn,
        m.male, my, mn,
        oth.others, othy, othn
    from(
    --healthy
        select eudract_number, hy.c as hy,  hn.c as hn,
                case when hy.c > hn.c then 1
                        when hy.c < hn.c then 0
                        when hy.c is null and hn.c is not null then 0
                        when hy.c is not null and hn.c is null then 1
                        else null end as healthy_volunteers
        from
            (select eudract_number, count(healthy_volunteers) as c
             from euctr_population
             where healthy_volunteers = 1
             group by eudract_number
             ) as hy
        full join
            (select eudract_number, count(healthy_volunteers) as c
             from euctr_population
             where healthy_volunteers = 0
             group by eudract_number
             ) as hn
        using (eudract_number)
    ) as hl
    join(
    --elderly
        select eudract_number, ely.c as ely,  eln.c as eln,
                case when ely.c > eln.c then 1
                        when ely.c < eln.c then 0
                        when ely.c is null and eln.c is not null then 0
                        when ely.c is not null and eln.c is null then 1
                        else null end as elderly
        from
            (select eudract_number, count(elderly) as c
             from euctr_population
             where elderly = 1
             group by eudract_number
             ) as ely
        full join
            (select eudract_number, count(elderly) as c
             from euctr_population
             where elderly = 0
             group by eudract_number
             ) as eln
        using (eudract_number)
    ) as el using (eudract_number)
    --adults
    join(
        select eudract_number,
                   ady.c as ady,  adn.c as adn, -- counts
                   case when ady.c > adn.c then 1
                        when ady.c < adn.c then 0
                        when ady.c is null and adn.c is not null then 0
                        when ady.c is not null and adn.c is null then 1
                        else null end as adults -- column name
        from(
            select eudract_number, count(adults) as c
            from euctr_population where adults = 1
            group by eudract_number
        ) as ady
        full join(
            select eudract_number, count(adults) as c
            from euctr_population where adults = 0
            group by eudract_number
        ) as adn  using (eudract_number)
--         where adn.c is not null -- cond
    ) as ad using (eudract_number)
    --adolescents
    join (
        select eudract_number,  adoy.c as adoy,  adon.c as adon, -- counts
                   case when adoy.c > adon.c then 1
                        when adoy.c < adon.c then 0
                        when adoy.c is null and adon.c is not null then 0
                        when adoy.c is not null and adon.c is null then 1
                        else null end as adolescents -- column name
        from(
            select eudract_number, count(adolescents) as c
            from euctr_population where adolescents = 1
            group by eudract_number
            ) as adoy
            full join (
                select eudract_number, count(adolescents) as c
                from euctr_population where adolescents = 0
                group by eudract_number
            ) as adon using (eudract_number)
    ) as ado using (eudract_number)
    --children
    join (
        select eudract_number,  chy.c as chy,  chn.c as chn, -- counts
                   case when chy.c > chn.c then 1
                        when chy.c < chn.c then 0
                        when chy.c is null and chn.c is not null then 0
                        when chy.c is not null and chn.c is null then 1
                        else null end as children -- column name
        from(
            select eudract_number, count(children) as c
            from euctr_population where children = 1
            group by eudract_number
            ) as chy
            full join (
                select eudract_number, count(children) as c
                from euctr_population where children = 0
                group by eudract_number
            ) as chn using (eudract_number)
--         where chy.c is not null -- cond
    ) as ch using (eudract_number)
    --female
    join (
        select eudract_number,  fy.c as fy,  fn.c as fn, -- counts
                   case when fy.c > fn.c then 1
                        when fy.c < fn.c then 0
                        when fy.c is null and fn.c is not null then 0
                        when fy.c is not null and fn.c is null then 1
                        else null end as female -- column name
        from(
            select eudract_number, count(female) as c
            from euctr_population where female = 1
            group by eudract_number
            ) as fy
            full join (
                select eudract_number, count(female) as c
                from euctr_population where female = 0
                group by eudract_number
            ) as fn using (eudract_number)
    ) as f using (eudract_number)
    --male
    join (
        select eudract_number,  my.c as my,  mn.c as mn, -- counts
                   case when my.c > mn.c then 1
                        when my.c < mn.c then 0
                        when my.c is null and mn.c is not null then 0
                        when my.c is not null and mn.c is null then 1
                        else null end as male -- column name
        from(
            select eudract_number, count(male) as c
            from euctr_population where male = 1
            group by eudract_number
            ) as my
            full join (
                select eudract_number, count(male) as c
                from euctr_population where male = 0
                group by eudract_number
            ) as mn using (eudract_number)
    ) as m using (eudract_number)
    --others
    join (
        select eudract_number,  othy.c as othy,  othn.c as othn, -- counts
                   case when othy.c > othn.c then 1
                        when othy.c < othn.c then 0
                        when othy.c is null and othn.c is not null then 0
                        when othy.c is not null and othn.c is null then 1
                        else null end as others -- column name
        from(
            select eudract_number, count(others) as c
            from euctr_population where others = 1
            group by eudract_number
            ) as othy
            full join (
                select eudract_number, count(others) as c
                from euctr_population where others = 0
                group by eudract_number
            ) as othn using (eudract_number)
    ) as oth using (eudract_number)
         )

select distinct g.*
from (select distinct g.eudract_number,
                      g.elderly, g.adults, g.adolescents, g.children,
                      g.female, g.male, g.others,
                      g.healthy_volunteers

          from gs_eudract g
          join euctr_population using (eudract_number)
          join eudract.population_description p using (eudract_number)
          where g.eudract_number is not null
            and p.inclusion is not null
            and g.elderly is not null
            and g.adults is not null
            and g.adolescents is not null
            and g.children is not null
            and g.female is not null
            and g.male is not null
            and g.others is not null
            and g.healthy_volunteers is not null
          ) as g
join ctgov c on c.eudract_number = g.eudract_number --38
    and c.elderly = g.elderly and c.adults = g.adults
    and c.adolescents = g.adolescents and c.children = g.children
    and c.male = g.male and c.female = g.female and c.others = g.others
    and c.healthy_volunteers = g.healthy_volunteers
 join drks d on c.eudract_number = d.eudract_number --79
    and c.elderly = d.elderly and c.adults = d.adults
    and c.adolescents = d.adolescents and c.children = d.children
    and c.male = d.male and c.female = d.female and c.others = d.others
order by g.eudract_number desc
;

