# lets understand ATTENTIONS
# suppose we have a model processing above sentence 
# when the model is processing cricket how can it decide which other words are important 

# --> attentions give model a mechanism to determine those relationship.

# ====================================================================================

# Causal Self-Attention

# This is especially important because you're learning LLMs/GPT-style models.
# Normal self-attention allows:
# the main thing here is they can allow look into the future tokens

# I → love
# I → playing
# I → cricket

# love → I
# love → playing
# love → cricket

# playing → I
# playing → love
# playing → cricket

# but when the gpr is generating a text it must not look into the future 
#  if the model is allowed to see the future it would be cheating 

# ===================================================================================

# CASUAL MASKING 

#           I   love  playing  cricket
# I         ✓    ✗      ✗        ✗
# love      ✓    ✓      ✗        ✗
# playing   ✓    ✓      ✓        ✗
# cricket   ✓    ✓      ✓        ✓

# This triangular strucutre is similar to GPT 
#  we will implwmwnt this using something like 

mask = torch.trill(torch.ones(T,T))  # ---> # this is used to prevent attention fro looking ahead



