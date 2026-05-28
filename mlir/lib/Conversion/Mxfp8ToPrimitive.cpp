/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PassDetail.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"
#include "allo/Support/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace allo;

namespace mlir {
namespace allo {
#define GEN_PASS_DEF_MXFP8TOPRIMITIVE
#include "allo/Conversion/Passes.h.inc"
} // namespace allo
} // namespace mlir

namespace mlir {
namespace allo {

static constexpr int64_t kE4M3Bias = 7;
static constexpr int64_t kE8M0Bias = 127;

static Value createConstI32(OpBuilder &builder, Location loc, int32_t v) {
  return builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                         builder.getI32IntegerAttr(v));
}

static Value createConstF32(OpBuilder &builder, Location loc, float v) {
  return builder.create<arith::ConstantOp>(loc, builder.getF32Type(),
                                           builder.getF32FloatAttr(v));
}

static Value castToI32(OpBuilder &builder, Location loc, Value v) {
  Type i32 = builder.getI32Type();
  if (v.getType() == i32)
    return v;
  if (v.getType().isF32())
    return builder.create<arith::FPToSIOp>(loc, i32, v);
  if (auto intTy = dyn_cast<IntegerType>(v.getType())) {
    if (intTy.getWidth() < 32)
      return builder.create<arith::ExtUIOp>(loc, i32, v);
    if (intTy.getWidth() > 32)
      return builder.create<arith::TruncIOp>(loc, i32, v);
  }
  return v;
}

static Value castToF32(OpBuilder &builder, Location loc, Value v) {
  if (v.getType().isF32())
    return v;
  if (v.getType().isF64())
    return builder.create<arith::TruncFOp>(loc, builder.getF32Type(), v);
  return builder.create<arith::SIToFPOp>(loc, builder.getF32Type(), v);
}

static Value castToIndex(OpBuilder &builder, Location loc, Value v) {
  Type idx = builder.getIndexType();
  if (v.getType() == idx)
    return v;
  return builder.create<arith::IndexCastOp>(loc, idx, castToI32(builder, loc, v));
}

static Value castToI8(OpBuilder &builder, Location loc, Value v) {
  Type i8 = builder.getIntegerType(8);
  if (v.getType() == i8)
    return v;
  if (isa<Mxfp8Type>(v.getType()))
    return builder.create<arith::BitcastOp>(loc, i8, v);
  return builder.create<arith::TruncIOp>(loc, i8, castToI32(builder, loc, v));
}

Type convertMxfp8MemRefOrScalar(Type t, MLIRContext *ctx) {
  if (auto memrefType = dyn_cast<MemRefType>(t)) {
    Type et = memrefType.getElementType();
    if (isa<Mxfp8Type>(et))
      return memrefType.clone(IntegerType::get(ctx, 8));
    return t;
  }
  if (isa<Mxfp8Type>(t))
    return IntegerType::get(ctx, 8);
  return t;
}

void updateMxfp8FunctionSignature(func::FuncOp &funcOp) {
  FunctionType functionType = funcOp.getFunctionType();
  SmallVector<Type, 4> newResultTypes;
  SmallVector<Type, 8> newArgTypes;
  for (Type t : functionType.getResults())
    newResultTypes.push_back(convertMxfp8MemRefOrScalar(t, funcOp.getContext()));
  for (Type t : functionType.getInputs())
    newArgTypes.push_back(convertMxfp8MemRefOrScalar(t, funcOp.getContext()));
  for (Block &block : funcOp.getBlocks()) {
    for (unsigned i = 0; i < block.getNumArguments(); ++i) {
      Type newType =
          convertMxfp8MemRefOrScalar(block.getArgument(i).getType(),
                                     funcOp.getContext());
      block.getArgument(i).setType(newType);
    }
  }
  FunctionType newFunctionType =
      FunctionType::get(funcOp.getContext(), newArgTypes, newResultTypes);
  funcOp.setFunctionType(newFunctionType);
}

void updateMxfp8Alloc(func::FuncOp &f) {
  f.walk([&](memref::AllocOp allocOp) {
    Type newType =
        convertMxfp8MemRefOrScalar(allocOp.getType(), f.getContext());
    if (newType != allocOp.getType())
      allocOp.getResult().setType(cast<MemRefType>(newType));
  });
  f.walk([&](memref::AllocaOp allocOp) {
    Type newType =
        convertMxfp8MemRefOrScalar(allocOp.getType(), f.getContext());
    if (newType != allocOp.getType())
      allocOp.getResult().setType(cast<MemRefType>(newType));
  });
}

static Value lowerDecodeE4m3(OpBuilder &builder, Location loc, Value input) {
  Value u = castToI32(builder, loc, castToI8(builder, loc, input));
  Value sign = builder.create<arith::ShRUIOp>(loc, u.getType(), u,
                                              createConstI32(builder, loc, 7));
  Value exp = builder.create<arith::AndIOp>(
      loc, builder.create<arith::ShRUIOp>(loc, u.getType(), u,
                                          createConstI32(builder, loc, 3)),
      createConstI32(builder, loc, 15));
  Value mant = builder.create<arith::AndIOp>(loc, u, createConstI32(builder, loc, 7));

  Value isNan = builder.create<arith::AndIOp>(
      loc,
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, exp,
                                    createConstI32(builder, loc, 15)),
      builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, mant,
                                    createConstI32(builder, loc, 7)));

  Value expF = castToF32(builder, loc, exp);
  Value mantF = castToF32(builder, loc, mant);
  Value subnormVal = builder.create<arith::MulFOp>(
      loc,
      builder.create<arith::DivFOp>(loc, mantF, createConstF32(builder, loc, 8.0f)),
      builder.create<math::PowFOp>(
          loc, createConstF32(builder, loc, 2.0f),
          createConstF32(builder, loc, float(1 - kE4M3Bias))));
  Value normBase = builder.create<arith::AddFOp>(
      loc, createConstF32(builder, loc, 1.0f),
      builder.create<arith::DivFOp>(loc, mantF, createConstF32(builder, loc, 8.0f)));
  Value normExp = builder.create<arith::SubFOp>(
      loc, expF, createConstF32(builder, loc, float(kE4M3Bias)));
  Value normVal = builder.create<arith::MulFOp>(
      loc, normBase,
      builder.create<math::PowFOp>(loc, createConstF32(builder, loc, 2.0f),
                                 normExp));
  Value isSubnorm = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, exp, createConstI32(builder, loc, 0));
  Value val = builder.create<arith::SelectOp>(loc, isSubnorm, subnormVal, normVal);
  val = builder.create<arith::SelectOp>(loc, isNan, createConstF32(builder, loc, 0.0f),
                                        val);
  Value negVal = builder.create<arith::SubFOp>(
      loc, createConstF32(builder, loc, 0.0f), val);
  Value isSign = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ne, sign, createConstI32(builder, loc, 0));
  return builder.create<arith::SelectOp>(loc, isSign, negVal, val);
}

static Value lowerDecodeE8m0(OpBuilder &builder, Location loc, Value input) {
  Value u = castToI32(builder, loc, castToI8(builder, loc, input));
  Value isZero = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, u, createConstI32(builder, loc, 0));
  Value isMax = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, u, createConstI32(builder, loc, 255));
  Value invalid = builder.create<arith::OrIOp>(loc, isZero, isMax);
  Value exp = builder.create<arith::SubFOp>(
      loc, castToF32(builder, loc, u),
      createConstF32(builder, loc, float(kE8M0Bias)));
  Value scale = builder.create<math::PowFOp>(
      loc, createConstF32(builder, loc, 2.0f), exp);
  return builder.create<arith::SelectOp>(loc, invalid,
                                         createConstF32(builder, loc, 0.0f),
                                         scale);
}

static Value lowerEncodeE4m3(OpBuilder &builder, Location loc, Value input) {
  Value f = castToF32(builder, loc, input);
  Value zero = createConstF32(builder, loc, 0.0f);
  Value isZero = builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, f, zero);

  Value signBit = builder.create<arith::SelectOp>(
      loc,
      builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, f, zero),
      createConstI32(builder, loc, 1), createConstI32(builder, loc, 0));
  Value absF = builder.create<arith::SelectOp>(
      loc,
      builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, f, zero),
      builder.create<arith::SubFOp>(loc, zero, f), f);

  Value packed = createConstI32(builder, loc, 0);
  Value unbiasedExp = createConstI32(builder, loc, -20);
  auto searchLoop = builder.create<scf::ForOp>(
      loc, createConstI32(builder, loc, -20), createConstI32(builder, loc, 16),
      createConstI32(builder, loc, 1), ValueRange({unbiasedExp}),
      [&](OpBuilder &b, Location l, Value e, ValueRange iterArgs) {
        Value e_i32 = castToI32(b, l, e);
        Value p = b.create<math::PowFOp>(
            l, createConstF32(b, l, 2.0f), castToF32(b, l, e_i32));
        Value cond = b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OGE, absF, p);
        Value nextExp = b.create<arith::SelectOp>(l, cond, e_i32, iterArgs[0]);
        b.create<scf::YieldOp>(l, ValueRange({nextExp}));
      });
  unbiasedExp = searchLoop.getResult(0);

  Value expField = builder.create<arith::AddIOp>(
      loc, unbiasedExp, createConstI32(builder, loc, kE4M3Bias));
  Value isSubnorm = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sle, expField, createConstI32(builder, loc, 0));
  expField = builder.create<arith::SelectOp>(
      loc, isSubnorm, createConstI32(builder, loc, 0), expField);

  Value divisorSub = builder.create<math::PowFOp>(
      loc, createConstF32(builder, loc, 2.0f),
      createConstF32(builder, loc, float(1 - kE4M3Bias)));
  Value mantSub = castToI32(
      builder, loc,
      builder.create<arith::AddFOp>(
          loc,
          builder.create<arith::MulFOp>(
              loc, builder.create<arith::DivFOp>(loc, absF, divisorSub),
              createConstF32(builder, loc, 8.0f)),
          createConstF32(builder, loc, 0.5f)));

  Value divisorNorm = builder.create<math::PowFOp>(
      loc, createConstF32(builder, loc, 2.0f), castToF32(builder, loc, unbiasedExp));
  Value mantNorm = castToI32(
      builder, loc,
      builder.create<arith::AddFOp>(
          loc,
          builder.create<arith::MulFOp>(
              loc,
              builder.create<arith::SubFOp>(
                  loc, builder.create<arith::DivFOp>(loc, absF, divisorNorm),
                  createConstF32(builder, loc, 1.0f)),
              createConstF32(builder, loc, 8.0f)),
          createConstF32(builder, loc, 0.5f)));

  Value mant = builder.create<arith::SelectOp>(loc, isSubnorm, mantSub, mantNorm);
  Value mantOverflow = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, mant, createConstI32(builder, loc, 8));
  mant = builder.create<arith::SelectOp>(
      loc, mantOverflow, createConstI32(builder, loc, 0), mant);
  expField = builder.create<arith::SelectOp>(
      loc, mantOverflow,
      builder.create<arith::AddIOp>(loc, expField, createConstI32(builder, loc, 1)),
      expField);

  Value saturated = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, expField, createConstI32(builder, loc, 15));
  expField = builder.create<arith::SelectOp>(
      loc, saturated, createConstI32(builder, loc, 14), expField);
  mant = builder.create<arith::SelectOp>(
      loc, saturated, createConstI32(builder, loc, 7), mant);

  packed = builder.create<arith::OrIOp>(
      loc,
      builder.create<arith::ShLIOp>(loc, signBit,
                                    createConstI32(builder, loc, 7)),
      builder.create<arith::OrIOp>(
          loc,
          builder.create<arith::ShLIOp>(loc, expField,
                                        createConstI32(builder, loc, 3)),
          mant));
  packed = builder.create<arith::SelectOp>(loc, isZero, createConstI32(builder, loc, 0),
                                         packed);
  return builder.create<arith::TruncIOp>(loc, builder.getIntegerType(8), packed);
}

static Value lowerEncodeE8m0(OpBuilder &builder, Location loc, Value input) {
  Value scale = castToF32(builder, loc, input);
  Value zero = createConstF32(builder, loc, 0.0f);
  Value isInvalid = builder.create<arith::CmpFOp>(
      loc, arith::CmpFPredicate::OLE, scale, zero);
  Value result = createConstI32(builder, loc, 0);
  auto searchLoop = builder.create<scf::ForOp>(
      loc, createConstI32(builder, loc, 1), createConstI32(builder, loc, 255),
      createConstI32(builder, loc, 1), ValueRange({createConstI32(builder, loc, 254)}),
      [&](OpBuilder &b, Location l, Value e, ValueRange iterArgs) {
        Value e_i32 = castToI32(b, l, e);
        Value p = b.create<math::PowFOp>(
            l, createConstF32(b, l, 2.0f),
            castToF32(b, l, b.create<arith::SubIOp>(
                                l, e_i32, createConstI32(b, l, kE8M0Bias))));
        Value cond = b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OGE, p, scale);
        Value isDefault = b.create<arith::CmpIOp>(
            l, arith::CmpIPredicate::eq, iterArgs[0], createConstI32(b, l, 254));
        Value picked = b.create<arith::SelectOp>(l, cond, e_i32, iterArgs[0]);
        Value next = b.create<arith::SelectOp>(l, isDefault, picked, iterArgs[0]);
        b.create<scf::YieldOp>(l, ValueRange({next}));
      });
  result = searchLoop.getResult(0);
  result = builder.create<arith::SelectOp>(loc, isInvalid, createConstI32(builder, loc, 0),
                                           result);
  return builder.create<arith::TruncIOp>(loc, builder.getIntegerType(8), result);
}

static int64_t getBlockSizeFromMemRef(Value memref) {
  auto ty = cast<MemRefType>(memref.getType());
  if (auto mx = dyn_cast<Mxfp8Type>(ty.getElementType()))
    return static_cast<int64_t>(mx.getBlockSize());
  if (ty.hasStaticShape() && !ty.getShape().empty())
    return ty.getShape()[0];
  return 32;
}

static void lowerDecodeMxfp8Block(DecodeMxfp8BlockOp op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  Value scale = lowerDecodeE8m0(builder, loc, op.getScale());
  int64_t bs = getBlockSizeFromMemRef(op.getData());
  Value c0 = createConstI32(builder, loc, 0);
  Value c1 = createConstI32(builder, loc, 1);
  Value cbs = createConstI32(builder, loc, bs);
  auto loop = builder.create<scf::ForOp>(
      loc, c0, cbs, c1, ValueRange(),
      [&](OpBuilder &b, Location l, Value i, ValueRange) {
        Value idx = castToIndex(b, l, i);
        Value dataVal = b.create<memref::LoadOp>(l, op.getData(), ValueRange({idx}));
        Value decoded = lowerDecodeE4m3(b, l, dataVal);
        Value prod = b.create<arith::MulFOp>(l, decoded, scale);
        b.create<memref::StoreOp>(l, prod, op.getOut(), ValueRange({idx}));
        b.create<scf::YieldOp>(l);
      });
  builder.setInsertionPointAfter(loop);
}

static void lowerEncodeMxfp8BlockImpl(OpBuilder &builder, Location loc, Value data,
                                      Value scaleOut, Value dataOut, int64_t bs) {
  Value c0 = createConstI32(builder, loc, 0);
  Value c1 = createConstI32(builder, loc, 1);
  Value cbs = createConstI32(builder, loc, bs);
  Value maxVal = createConstF32(builder, loc, 0.0f);

  auto maxLoop = builder.create<scf::ForOp>(
      loc, c0, cbs, c1, ValueRange({maxVal}),
      [&](OpBuilder &b, Location l, Value i, ValueRange iterArgs) {
        Value idx = castToIndex(b, l, i);
        Value v = b.create<memref::LoadOp>(l, data, ValueRange({idx}));
        Value av = b.create<arith::SelectOp>(
            l, b.create<arith::CmpFOp>(l, arith::CmpFPredicate::OLT, v,
                                       createConstF32(b, l, 0.0f)),
            b.create<arith::SubFOp>(l, createConstF32(b, l, 0.0f), v), v);
        Value next = b.create<arith::MaximumFOp>(l, iterArgs[0], av);
        b.create<scf::YieldOp>(l, ValueRange({next}));
      });
  maxVal = maxLoop.getResult(0);

  Value scaleByte = lowerEncodeE8m0(builder, loc, maxVal);
  builder.create<memref::StoreOp>(loc, scaleByte, scaleOut, ValueRange({castToIndex(builder, loc, c0)}));
  Value scale = lowerDecodeE8m0(builder, loc, scaleByte);

  auto encLoop = builder.create<scf::ForOp>(
      loc, c0, cbs, c1, ValueRange(),
      [&](OpBuilder &b, Location l, Value i, ValueRange) {
        Value idx = castToIndex(b, l, i);
        Value v = b.create<memref::LoadOp>(l, data, ValueRange({idx}));
        Value scaled = b.create<arith::DivFOp>(l, v, scale);
        Value isZeroScale = b.create<arith::CmpFOp>(
            l, arith::CmpFPredicate::OEQ, scale, createConstF32(b, l, 0.0f));
        scaled = b.create<arith::SelectOp>(l, isZeroScale, v, scaled);
        Value encoded = lowerEncodeE4m3(b, l, scaled);
        b.create<memref::StoreOp>(l, encoded, dataOut, ValueRange({idx}));
        b.create<scf::YieldOp>(l);
      });
  builder.setInsertionPointAfter(encLoop);
}

static void lowerEncodeMxfp8Block(EncodeMxfp8BlockOp op) {
  OpBuilder builder(op);
  int64_t bs = getBlockSizeFromMemRef(op.getDataOut());
  lowerEncodeMxfp8BlockImpl(builder, op.getLoc(), op.getData(), op.getScaleOut(),
                            op.getDataOut(), bs);
}

static void lowerBlockAddMxfp8(BlockAddMxfp8Op op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  int64_t bs = getBlockSizeFromMemRef(op.getDataA());
  auto f32Ty = builder.getF32Type();
  auto bufTy = MemRefType::get({bs}, f32Ty);
  Value buf = builder.create<memref::AllocaOp>(loc, bufTy);
  Value sa = lowerDecodeE8m0(builder, loc, op.getScaleA());
  Value sb = lowerDecodeE8m0(builder, loc, op.getScaleB());
  Value c0 = createConstI32(builder, loc, 0);
  Value c1 = createConstI32(builder, loc, 1);
  Value cbs = createConstI32(builder, loc, bs);

  auto addLoop = builder.create<scf::ForOp>(
      loc, c0, cbs, c1, ValueRange(),
      [&](OpBuilder &b, Location l, Value i, ValueRange) {
        Value idx = castToIndex(b, l, i);
        Value a = lowerDecodeE4m3(
            b, l, b.create<memref::LoadOp>(l, op.getDataA(), ValueRange({idx})));
        Value bb = lowerDecodeE4m3(
            b, l, b.create<memref::LoadOp>(l, op.getDataB(), ValueRange({idx})));
        Value sum = b.create<arith::AddFOp>(
            l, b.create<arith::MulFOp>(l, a, sa),
            b.create<arith::MulFOp>(l, bb, sb));
        b.create<memref::StoreOp>(l, sum, buf, ValueRange({idx}));
        b.create<scf::YieldOp>(l);
      });
  builder.setInsertionPointAfter(addLoop);
  lowerEncodeMxfp8BlockImpl(builder, loc, buf, op.getScaleOut(), op.getDataOut(), bs);
}

static void lowerBlockMatMulMxfp8(BlockMatMulMxfp8Op op) {
  OpBuilder builder(op);
  Location loc = op.getLoc();
  int64_t bs = getBlockSizeFromMemRef(op.getDataA());
  Value sa = lowerDecodeE8m0(builder, loc, op.getScaleA());
  Value sb = lowerDecodeE8m0(builder, loc, op.getScaleB());
  Value acc = createConstF32(builder, loc, 0.0f);
  Value c0 = createConstI32(builder, loc, 0);
  Value c1 = createConstI32(builder, loc, 1);
  Value cbs = createConstI32(builder, loc, bs);

  auto dotLoop = builder.create<scf::ForOp>(
      loc, c0, cbs, c1, ValueRange({acc}),
      [&](OpBuilder &b, Location l, Value i, ValueRange iterArgs) {
        Value idx = castToIndex(b, l, i);
        Value a = lowerDecodeE4m3(
            b, l, b.create<memref::LoadOp>(l, op.getDataA(), ValueRange({idx})));
        Value bb = lowerDecodeE4m3(
            b, l, b.create<memref::LoadOp>(l, op.getDataB(), ValueRange({idx})));
        Value prod = b.create<arith::MulFOp>(
            l, b.create<arith::MulFOp>(l, a, sa),
            b.create<arith::MulFOp>(l, bb, sb));
        Value next = b.create<arith::AddFOp>(l, iterArgs[0], prod);
        b.create<scf::YieldOp>(l, ValueRange({next}));
      });
  acc = dotLoop.getResult(0);
  builder.setInsertionPointAfter(dotLoop);

  auto f32Ty = builder.getF32Type();
  auto outBufTy = MemRefType::get({1}, f32Ty);
  Value outBuf = builder.create<memref::AllocaOp>(loc, outBufTy);
  builder.create<memref::StoreOp>(loc, acc, outBuf, ValueRange({castToIndex(builder, loc, c0)}));
  lowerEncodeMxfp8BlockImpl(builder, loc, outBuf, op.getScaleOut(), op.getDataOut(), 1);
}

void visitMxfp8Operation(Operation &op);
void visitMxfp8Region(Region &region);
void visitMxfp8Block(Block &block);

void visitMxfp8Operation(Operation &op) {
  if (auto mxOp = dyn_cast<DecodeE4m3Op>(op)) {
    OpBuilder builder(mxOp);
    Value res = lowerDecodeE4m3(builder, mxOp.getLoc(), mxOp.getInput());
    mxOp.getResult().replaceAllUsesWith(res);
  } else if (auto mxOp = dyn_cast<EncodeE4m3Op>(op)) {
    OpBuilder builder(mxOp);
    Value res = lowerEncodeE4m3(builder, mxOp.getLoc(), mxOp.getInput());
    mxOp.getResult().replaceAllUsesWith(res);
  } else if (auto mxOp = dyn_cast<DecodeE8m0Op>(op)) {
    OpBuilder builder(mxOp);
    Value res = lowerDecodeE8m0(builder, mxOp.getLoc(), mxOp.getInput());
    mxOp.getResult().replaceAllUsesWith(res);
  } else if (auto mxOp = dyn_cast<EncodeE8m0Op>(op)) {
    OpBuilder builder(mxOp);
    Value res = lowerEncodeE8m0(builder, mxOp.getLoc(), mxOp.getInput());
    mxOp.getResult().replaceAllUsesWith(res);
  } else if (auto mxOp = dyn_cast<DecodeMxfp8BlockOp>(op)) {
    lowerDecodeMxfp8Block(mxOp);
  } else if (auto mxOp = dyn_cast<EncodeMxfp8BlockOp>(op)) {
    lowerEncodeMxfp8Block(mxOp);
  } else if (auto mxOp = dyn_cast<BlockAddMxfp8Op>(op)) {
    lowerBlockAddMxfp8(mxOp);
  } else if (auto mxOp = dyn_cast<BlockMatMulMxfp8Op>(op)) {
    lowerBlockMatMulMxfp8(mxOp);
  }
}

void visitMxfp8Block(Block &block) {
  SmallVector<Operation *, 10> mxOps;
  for (Operation &op : block.getOperations()) {
    if (llvm::isa<DecodeE4m3Op, EncodeE4m3Op, DecodeE8m0Op, EncodeE8m0Op,
                  DecodeMxfp8BlockOp, EncodeMxfp8BlockOp, BlockAddMxfp8Op,
                  BlockMatMulMxfp8Op>(op))
      mxOps.push_back(&op);
  }
  for (Operation *op : mxOps)
    visitMxfp8Operation(*op);
  for (Operation *op : mxOps)
    op->erase();
}

void visitMxfp8Region(Region &region) {
  for (auto &block : region.getBlocks())
    visitMxfp8Block(block);
}

bool applyMxfp8ToPrimitive(ModuleOp &mod) {
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    updateMxfp8FunctionSignature(func);
    updateMxfp8Alloc(func);
    visitMxfp8Region(func.getBody());
  }
  return true;
}

} // namespace allo
} // namespace mlir

namespace {

struct AlloMxfp8ToPrimitiveTransformation
    : public mlir::allo::impl::Mxfp8ToPrimitiveBase<
          AlloMxfp8ToPrimitiveTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyMxfp8ToPrimitive(mod))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createMxfp8ToPrimitivePass() {
  return std::make_unique<AlloMxfp8ToPrimitiveTransformation>();
}

} // namespace allo
} // namespace mlir
