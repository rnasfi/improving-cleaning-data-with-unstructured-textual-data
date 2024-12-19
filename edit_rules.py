trials_rules = [
                {#0
                    'open' : 1,
                    'double_blind' : 1
                },
                {#1
                    'single_blind' : 1,
                    'double_blind' : 1
                },
                {#2
                    'open' : 1,
                    'single_blind' : 1
                },
                {#3
                    'open' : 0,
                    'single_blind' : 0,
                    'double_blind' : 0
                } ,
                # labels are encoded
                {#0
                    'controlled' : 0,
                    'placebo' : 1
                },
                {#1
                    'controlled' : 0,
                    'active_comparator' : 1
                },
                {#6
                    'controlled' : 1,
                    'arms' : 0
                },
                {#2
                    'placebo' : 1,
                    'arms' : 0
                },
                {#3
                    'arms' : 0,
                    'active_comparator' : 1
                },
                {#4
                    'placebo' : 1,
                    'arms' : 1
                },
                {#5
                    'arms' : 1,
                    'active_comparator' : 1
                }, 
                {#0
                    'parallel_group' : 1,
                    'crossover' : 1
                },
               # {'arms' : 3}                
             ]
trials_masking = [
                {#0
                    'open' : 1,
                    'double_blind' : 1
                },
                {#1
                    'single_blind' : 1,
                    'double_blind' : 1
                },
                {#2
                    'open' : 1,
                    'single_blind' : 1
                },
                {#3
                    'open' : 0,
                    'single_blind' : 0,
                    'double_blind' : 0
                }    
    
             ]

trials_control = [ # labels are encoded
                {#0
                    'controlled' : 0,
                    'placebo' : 1
                },
                {#1
                    'controlled' : 0,
                    'active_comparator' : 1
                },
                {#6
                    'controlled' : 1,
                    'arms' : 0
                },
                {#2
                    'placebo' : 1,
                    'arms' : 0
                },
                {#3
                    'arms' : 0,
                    'active_comparator' : 1
                },
                {#4
                    'placebo' : 1,
                    'arms' : 1
                },
                {#5
                    'arms' : 1,
                    'active_comparator' : 1
                }
                ]


clinical_masking = [
                {#0
                    'open' : 'Yes',
                    'double_blind' : 'Yes'
                },
                {#1
                    'single_blind' : 'Yes',
                    'double_blind' : 'Yes'
                },
                {#2
                    'open' : 'Yes',
                    'single_blind' : 'Yes'
                },
                {#3
                    'open' : 'No',
                    'single_blind' : 'No',
                    'double_blind' : 'No'
                }]

# crossover and parallel cannot occur simultaneously
trials_assign = [
                {#0
                    'parallel_group' : 'Yes',
                    'crossover' : 'Yes'
                }
]


def check_comparator(row):
    violated = False
    if row['active_comparator'] == 1 and (row['arms'] in [1, 0]):
        violated = True
    if row['placebo'] == 1 and (row['arms'] in [1, 0]):
        violated = True
    return  violated

def check(row, edit_rules):  
    
    violated = False
    
    for i in range(len(edit_rules)):
        
        edit_rule = edit_rules[i]

        # if two attributes have the same values as defined in the set of rules
        if all(row[attr] == edit_rule[attr] for attr in edit_rule.keys()): 
            violated = True

    return violated


def get_violated_rows(dataset, edit_rules):
    violations = {}

    for index, row in dataset.iterrows():
        
        for i in range(len(edit_rules)):
            
            edit_rule = edit_rules[i]
                        
            if all(row[attr] == edit_rule[attr] for attr in edit_rule.keys()):#keys are: double, single,..
                print("violated", edit_rule) #edit rule here is violated

                #violations has the format {1: [list/nb of violated rules]}
                if index not in violations:
                    violations[index] = [i]                    
                else:
                    violations[index].append(i) #when a row already has violated a previous rule, we add the current on to the list 
    return violations
