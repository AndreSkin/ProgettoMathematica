from typing import List

def gf_mult(x: int, y: int, prim=0x11d, field_charac_full=256, carryless=True) -> int:
    '''Galois Field integer multiplication using Russian Peasant Multiplication algorithm (faster than the standard multiplication + modular reduction).'''
    r = 0
    while y:
        if y & 1:
            r = r ^ x if carryless else r + x
        y = y >> 1
        x = x << 1
        if prim > 0 and x & field_charac_full:
            x = x ^ prim
    return r

def gf_poly_scale(p: List[int], x: int) -> List[int]:
    '''Multiplies a polynomial in GF(2^p) by a scalar.'''
    return [gf_mult(p[i], x) for i in range(len(p))]

def gf_poly_add(p: List[int], q: List[int]) -> List[int]:
    '''Adds two polynomials in GF(2^p).'''
    r = [0] * max(len(p), len(q))
    for i in range(len(p)):
        r[i+len(r)-len(p)] = p[i]
    for i in range(len(q)):
        r[i+len(r)-len(q)] ^= q[i]
    return r

def gf_poly_mult(p: List[int], q: List[int]) -> List[int]:
    '''Multiplies two polynomials in GF(2^p).'''
    r = [0] * (len(p)+len(q)-1)
    for j in range(len(q)):
        for i in range(len(p)):
            r[i+j] ^= gf_mult(p[i], q[j])
    return r

def gf_poly_eval(poly: List[int], x: int) -> int:
    '''Evaluates a polynomial in GF(2^p) at a point x.'''
    y = poly[0]
    for i in range(1, len(poly)):
        y = gf_mult(y, x) ^ poly[i]
    return y

def rs_generator_poly(nsym: int) -> List[int]:
    '''Generates the Reed-Solomon generator polynomial of nSym symbols.'''
    g = [1]
    for i in range(nsym):
        g = gf_poly_mult(g, [1, i])
    return g

def rs_encode_msg(msg_in: List[int], nsym: int) -> List[int]:
    '''Encodes a message using Reed-Solomon.'''
    if (len(msg_in) + nsym) > 255:
        raise ValueError("Message is too long")
    
    gen = rs_generator_poly(nsym)
    
    msg_out = [0] * (len(msg_in) + nsym)
    
    msg_out[:len(msg_in)] = msg_in
    
    for i in range(len(msg_in)):
        coef = msg_out[i]
        
        if coef != 0:
            for j in range(1, len(gen)):
                msg_out[i+j] ^= gf_mult(gen[j], coef)
    
    msg_out[:len(msg_in)] = msg_in
    
    return msg_out

def rs_decode_msg(msg_in: List[int], nsym: int) -> List[int]:
    '''Decodes a message using Reed-Solomon.'''
    
    if (len(msg_in) > 255):
        raise ValueError("Message is too long")
    
    # Compute the syndromes polynomial
    synd = [0] * nsym
    error = False
    for i in range(nsym):
        tmp = gf_poly_eval(msg_in[::-1], i)
        
        if tmp != 0:
            error = True
        
        synd[i] = tmp
    
    if not error:
        return msg_in[:-nsym]
    
    # Compute the error locator polynomial using Berlekamp-Massey algorithm
    err_loc = [1]
    
    old_loc = [1]
    
    for i in range(nsym):
        
        old_loc.append(0)
        
        delta = synd[i]
        
        for j in range(1, len(err_loc)):
            delta ^= gf_mult(err_loc[-(j+1)], synd[i-j])
        
        if delta != 0:
            
            if len(old_loc) > len(err_loc):
                
                new_loc = gf_poly_scale(old_loc, delta)
                old_loc = gf_poly_scale(err_loc, gf_mult(delta, 255))
                err_loc = new_loc
            
            err_loc = gf_poly_add(err_loc, gf_poly_scale(old_loc, delta))
    
    errs = len(err_loc) - 1
    
    if errs * 2 > nsym:
        raise ValueError("Too many errors to correct")
    
    # Find the roots of the error locator polynomial (Chien search)
    err_pos = []
    
    for i in range(255):
        if gf_poly_eval(err_loc[::-1], i) == 0:
            err_pos.append(255 - i - 1)
    
    if len(err_pos) != errs:
        return None
    
    # Compute the error evaluator polynomial using Forney algorithm
    err_eval_reversed = [0] * (errs + 1)
    
    for i in range(errs + 1):
        tmp = 0
        
        for j in range(i + 1):
            tmp ^= gf_mult(synd[j], err_loc[errs - i + j])
        
        err_eval_reversed[i] = tmp
    
    # Compute the error values and apply them to the message
    msg_out = list(msg_in)
    
    for i in range(len(err_pos)):
        
        pos = err_pos[i]
        
        x_inv = gf_mult(pos, 255)
        
        numerator = gf_poly_eval(err_eval_reversed[::-1], x_inv)
        
        denominator = 1
        
        for j in range(errs):
            if j != i:
                denominator ^= gf_mult(1 ^ pos ^ err_pos[j], x_inv)
        
        msg_out[pos] ^= gf_mult(numerator, denominator ^ 255)
    
    return msg_out[:-nsym]