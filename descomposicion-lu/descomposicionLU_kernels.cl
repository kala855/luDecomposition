/*
 * Application descomposicionLU
 * Copyright (C) John H. Osorio-Ríos 2013 <john@arwen>
 * 
descomposicionLU is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * descomposicionLU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

__kernel void descomposicionluKernel(__global  unsigned int * output,
                             __global  unsigned int * input,
                             const     unsigned int multiplier)
{
    uint tid = get_global_id(0);
    
    output[tid] = input[tid] * multiplier;
}
