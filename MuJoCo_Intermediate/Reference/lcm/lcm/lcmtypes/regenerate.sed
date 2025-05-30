# This file is used by regenerate.sh to edit the lcm-gen output files.

# Edit the include paths.
s|<lcm/lcm_coretypes.h>|"../lcm_coretypes.h"\n#include "../lcm_export.h"|g;

# Edit all function declarations to flag as non-exported. The following regular
# expressions match the start-of-line (i.e., return type) for all functions.
# (This also edits the *definitions* in the .c files, but that's harmless.)
s|^\(void\) |LCM_NO_EXPORT\n\1 |g;
s|^\(int\) |LCM_NO_EXPORT\n\1 |g;
s|^\(int64_t\) |LCM_NO_EXPORT\n\1 |g;
s|^\(uint64_t\) |LCM_NO_EXPORT\n\1 |g;
s|^\(channel[^ ]*_t *\*\)|LCM_NO_EXPORT\n\1|g;

# Edit the codegen disclaimer comment.
s|Generated by lcm-gen.*|Generated by the regenerate.sh script in this directory|g;
