// Port of Instructions/InstEmitException.cs

use crate::decoder::opcode_kind::OpCodeKind;

use super::context::ArmEmitterCtx;
use crate::instructions::native_function::NativeFn;

// C# casts CurrOp to OpCodeException to read Id.
fn exception_id(ctx: &ArmEmitterCtx) -> u16 {
    match ctx.curr_op().kind {
        OpCodeKind::Exception { id } => id,
        _ => panic!("expected Exception opcode, got {:?}", ctx.curr_op().kind),
    }
}

pub fn emit_brk(ctx: &mut ArmEmitterCtx) {
    let address = ctx.curr_op().address;
    let id = exception_id(ctx);

    ctx.store_to_context();

    let addr_op = ctx.ctx.operands.constant_i64(address as i64);
    let id_op = ctx.ctx.operands.constant_i32(id as i32);
    ctx.call(NativeFn::Break, &[addr_op, id_op]);

    ctx.load_from_context();

    let ret = ctx.ctx.operands.constant_i64(address as i64);
    ctx.return_with(ret);
}

pub fn emit_svc(ctx: &mut ArmEmitterCtx) {
    let address = ctx.curr_op().address;
    let id = exception_id(ctx);

    ctx.store_to_context();

    let addr_op = ctx.ctx.operands.constant_i64(address as i64);
    let id_op = ctx.ctx.operands.constant_i32(id as i32);
    ctx.call(NativeFn::SupervisorCall, &[addr_op, id_op]);

    ctx.load_from_context();

    ctx.emit_synchronization();
}

// C# Und reads only the base OpCode (Address + RawOpCode)  no kind
// constraint. Not in the A64 table: C# wires it as the fallback for
// undefined encodings; the Rust decoder returns DecodeError::Undefined
// instead, and the block decoder maps that to this emitter.
pub fn emit_und(ctx: &mut ArmEmitterCtx) {
    let address = ctx.curr_op().address;
    let raw = ctx.curr_op().raw;

    ctx.store_to_context();

    let addr_op = ctx.ctx.operands.constant_i64(address as i64);
    let raw_op = ctx.ctx.operands.constant_i32(raw as i32);
    ctx.call(NativeFn::Undefined, &[addr_op, raw_op]);

    ctx.load_from_context();

    let ret = ctx.ctx.operands.constant_i64(address as i64);
    ctx.return_with(ret);
}
