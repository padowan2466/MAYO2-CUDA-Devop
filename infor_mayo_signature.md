  ### Function Parameters                                                                                      
                                                                                                               
    int mayo_sign_signature(const mayo_params_t *p, unsigned char *sig,                                        
                            size_t *siglen, const unsigned char *m, size_t mlen,                               
                            const unsigned char *csk)                                                          
                                                                                                               
  •  p  (const  mayo_params_t * ): A pointer to the MAYO parameters structure, which contains dimensions like  
  $m$, $n$, $o$, $k$, $v$, and specific lengths for keys and signatures.                                       
  •  sig  ( unsigned char * ): Pointer to the output buffer where the generated signature will be written. The 
  buffer must be at least  PARAM_sig_bytes(p)  bytes.                                                          
  •  siglen  ( size_t * ): Pointer to a  size_t  variable that will be populated with the final length of the  
  written signature.                                                                                           
  •  m  (const  unsigned char * ): Pointer to the message byte array to be signed.                             
  •  mlen  ( size_t ): The length of the message  m  in bytes.                                                 
  •  csk  (const  unsigned char * ): Pointer to the compressed secret key.                                     
  ──────                                                                                                       
  ### Step-by-Step Execution of  mayo_sign_signature                                                           
                                                                                                               
  #### Step 1: Secret Key Expansion                                                                            
                                                                                                               
    ret = mayo_expand_sk(p, csk, &sk);                                                                         
                                                                                                               
  • What it does: Expands the compact/compressed secret key  csk  (which stores the seed) into its full        
  structure  sk , exposing the matrices $P_1$ (quadratic component), $L$ (linear component), and $O$ (the      
  secret oil space matrix).                                                                                    
                                                                                                               
  #### Step 2: Hashing the Message                                                                             
                                                                                                               
    shake256(tmp, param_digest_bytes, m, mlen);                                                                
                                                                                                               
  • What it does: Hashes the input message  m  of length  mlen  using SHAKE256 to produce a message digest of  
  length  param_digest_bytes . The digest is stored at the beginning of the  tmp  buffer.                      
                                                                                                               
  #### Step 3: Salt Generation                                                                                 
                                                                                                               
    // Note: In this specific implementation, a fixed seed is used for reproducibility                         
    for (int i = 0; i < param_salt_bytes; i++) {                                                               
      tmp[param_digest_bytes + i] = 1;                                                                         
    }                                                                                                          
    memcpy(tmp + param_digest_bytes + param_salt_bytes, seed_sk, param_sk_seed_bytes);                         
    shake256(salt, param_salt_bytes, tmp, param_digest_bytes + param_salt_bytes + param_sk_seed_bytes);        
                                                                                                               
  • What it does: Generates a signature salt. It combines the message digest, a salt seed (which is fixed to  1
  s here for debugging/reproducibility instead of  randombytes() ), and the secret key seed, hashing them      
  together via SHAKE256 to get the final  salt .                                                               
                                                                                                               
  #### Step 4: Computing Target Vector $t$                                                                     
                                                                                                               
    memcpy(tmp + param_digest_bytes, salt, param_salt_bytes);                                                  
    shake256(tenc, param_m_bytes, tmp, param_digest_bytes + param_salt_bytes);                                 
    decode(tenc, t, param_m);                                                                                  
                                                                                                               
  • What it does: Hashes the message digest and the salt together to generate an encoded target byte array     
  tenc . Then,  decode()  transforms  tenc  into the target vector $t \in \mathbb{F}_q^m$.                     
  ──────                                                                                                       
  #### Step 5: The Solving Loop (Finding a Solution in the Oil Space)                                          
                                                                                                               
  The algorithm enters a loop iterating  ctr  from $0$ to $255$ trying to find a valid signature solution. In  
  each iteration:                                                                                              
                                                                                                               
  1. Pseudorandom Generation of $V$:                                                                           
    *ctrbyte = (unsigned char)ctr;                                                                             
    shake256(V, param_k * param_v_bytes + param_r_bytes, tmp, ...);                                            
  Generates a pseudorandom vector $V$ (representing $k$ vinegar vectors $v_1, \dots, v_k$) and a randomizer    
  vector $r$ based on the message, salt, secret key seed, and current counter  ctr .                           
  2. Decoding Vinegar Vectors:                                                                                 
    for (int i = 0; i <= param_k - 1; ++i) {                                                                   
      decode(V + i * param_v_bytes, Vdec + i * param_v, param_v);                                              
    }                                                                                                          
  Decodes the generated bytes in $V$ into the vinegar vectors $v_i \in \mathbb{F}_q^v$ for each of the $k$     
  components.                                                                                                  
  3. Computing Internal Bilinear Forms:                                                                        
    compute_M_and_VPV(p, Vdec, L, P1, Mtmp, (uint64_t *)A);                                                    
  Computes the matrices $M_i$ and the bilinear evaluations $v_i^\top P_1 v_j$ using the secret key matrices    
  $P_1$, $L$ and the decoded vinegar vectors.                                                                  
  4. Right-Hand Side (RHS) Vector $y$ Calculation:                                                             
    compute_rhs(p, (uint64_t *)A, t, y);                                                                       
  Calculates the right-hand side vector $y \in \mathbb{F}_q^m$ of the linear system to be solved, representing 
  the target values after accounting for pure vinegar evaluations.                                             
  5. Linear System Matrix $A$ Construction:                                                                    
    compute_A(p, Mtmp, A);                                                                                     
    for (int i = 0; i < param_m; i++) {                                                                        
      A[(1 + i) * (param_k * param_o + 1) - 1] = 0;
    }
  Builds the system matrix $A$ representing the linear system of equations over the oil variables.             
  6. Target Vector $r$ Decoding:
    decode(V + param_k * param_v_bytes, r, param_k * param_o);
  Decodes the final part of $V$ into $r \in \mathbb{F}_q^{ko}$ to serve as a randomizer for sampling solutions.
  7. Solving $Ax = y$:
    if (sample_solution(p, A, y, r, x, param_k, param_o, param_m, param_A_cols)) {
      sol_found = 1;
      break;
    }
  Attempts to find a solution $x \in \mathbb{F}_q^{ko}$ to the system $Ax = y$. If a solution is found, it sets
  sol_found = 1  and exits the loop. If not, it clears temporary matrices and retries with the next  ctr  value.
  ──────
  #### Step 6: Mapping back to N-dimensional Space
  
  Once the oil space solution $x$ is found:
  
    for (int i = 0; i <= param_k - 1; ++i) {
      vi = Vdec + i * (param_n - param_o);
      mat_mul(sk.O, x + i * param_o, Ox, param_o, param_n - param_o, 1);
      mat_add(vi, Ox, s + i * param_n, param_n - param_o, 1);
      memcpy(s + i * param_n + (param_n - param_o), x + i * param_o, param_o);
    }
  
  • What it does: Maps the $v$-dimensional vinegar variables ($v_i$) and the $o$-dimensional oil solution      
  variables ($x_i$) back to the $n$-dimensional signature space. It performs the matrix transformation $s_i =  
  (v_i + O x_i) \parallel x_i$ for each of the $k$ components.
  
  #### Step 7: Encoding and Pack the Signature
  
    encode(s, sig, param_n * param_k);
    memcpy(sig + param_sig_bytes - param_salt_bytes, salt, param_salt_bytes);
    *siglen = param_sig_bytes;
  
  • What it does: Encodes the signature vector $s$ into bytes and writes it to the output  sig . Then, it      
  appends the  salt  at the end of the signature and updates the total written signature length in  *siglen .  
  
  #### Step 8: Secure Memory Clearing
  
    err:
      mayo_secure_clear(V, sizeof(V));
      mayo_secure_clear(Vdec, sizeof(Vdec));
      ...
      return ret;
  
  • What it does: Before exiting, the function securely zeroes out all stack variables and memory regions that 
  stored sensitive secret data (such as the secret key  sk , vinegar vectors, system matrix  A , and oil       
  variables $x$) to prevent side-channel leaks. It then returns the status code  ret .

